#!/usr/bin/env python3
"""DGX Spark native spiking language-model trainer.

This is a separate capability track from Stage 2e.  It optimizes next-token
cross entropy with surrogate gradients and uses parallel, non-reset LIF
integrators so the hot path maps to Blackwell tensor operations.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        return (membrane > 0).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (membrane,) = ctx.saved_tensors
        return grad_output / (1.0 + membrane.abs()).square()


spike = SurrogateSpike.apply


class ParallelLIF(nn.Module):
    """Causal leaky integration evaluated as a parallel prefix sum.

    Reset-free LIF is intentional in the first Spark gate: unlike a Python
    time-step loop it exposes the full sequence to CUDA and Tensor Core
    scheduling.  Four fixed time constants provide short/long memory bands.
    """

    def __init__(self, width: int, threshold: float = 1.0):
        super().__init__()
        bands = torch.tensor([0.90, 0.95, 0.98, 0.99], dtype=torch.float32)
        beta = bands.repeat_interleave((width + 3) // 4)[:width]
        self.register_buffer("beta", beta)
        self.threshold = nn.Parameter(torch.full((width,), threshold))

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        # Integration stays FP32 for long-time-constant numerical stability;
        # the surrounding projections remain BF16.
        length = current.shape[1]
        t = torch.arange(length, device=current.device, dtype=torch.float32)
        powers = self.beta[:, None].pow(t[None, :]).transpose(0, 1)
        membrane = torch.cumsum(current.float() / powers[None, :, :], dim=1)
        membrane = membrane * powers[None, :, :]
        return spike(membrane - self.threshold[None, None, :]).to(current.dtype)


class SpikingBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.lif = ParallelLIF(d_ff)
        self.out_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        spikes = self.lif(current)
        x = x + self.dropout(self.out_proj(spikes * torch.sigmoid(gate)))
        return x, spikes.mean()


class SparkSNNLM(nn.Module):
    def __init__(self, vocab: int, seq_len: int, d_model: int, d_ff: int,
                 layers: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(seq_len, d_model))
        self.blocks = nn.ModuleList(
            [SpikingBlock(d_model, d_ff, dropout) for _ in range(layers)]
        )
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_embedding(tokens) + self.position_embedding[:tokens.shape[1]]
        rates = []
        for block in self.blocks:
            x, rate = block(x)
            rates.append(rate)
        return self.lm_head(self.norm(x)), torch.stack(rates).mean()


def build_tokenizer(corpus: Path, output: Path, vocab_size: int) -> Tokenizer:
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>", "<bos>", "<eos>"],
        show_progress=True,
    )
    tokenizer.train([str(corpus)], trainer)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.save(str(output))
    return tokenizer


def load_tokens(corpus: Path, tokenizer: Tokenizer, tokenizer_path: Path,
                cache_path: Path | None = None) -> torch.Tensor:
    corpus_stat = corpus.stat()
    metadata = {
        "corpus_path": str(corpus.resolve()),
        "corpus_size": corpus_stat.st_size,
        "corpus_mtime_ns": corpus_stat.st_mtime_ns,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_size": tokenizer_path.stat().st_size,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "vocab": tokenizer.get_vocab_size(),
    }
    metadata_path = (
        cache_path.with_suffix(cache_path.suffix + ".json")
        if cache_path else None
    )
    if cache_path and metadata_path and cache_path.exists() and metadata_path.exists():
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        token_count = int(cached_metadata.pop("token_count", 0))
        if (
            cached_metadata == metadata
            and token_count >= 1024
            and cache_path.stat().st_size == token_count * 4
        ):
            tokens = torch.from_file(
                str(cache_path), shared=False, size=token_count, dtype=torch.int32
            )
            if tokens.numel() == token_count:
                print(
                    f"RUN token_cache=hit path={cache_path} "
                    f"tokens={tokens.numel()}",
                    flush=True,
                )
                return tokens
        print(f"RUN token_cache=stale path={cache_path}", flush=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_data = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        token_count = 0
        encoded_characters = 0
        try:
            with corpus.open(
                "r", encoding="utf-8", errors="replace"
            ) as source, temporary_data.open("wb") as destination:
                while chunk := source.read(4 * 1024 * 1024):
                    ids = tokenizer.encode(chunk).ids
                    array("i", ids).tofile(destination)
                    token_count += len(ids)
                    encoded_characters += len(chunk)
                    if encoded_characters // (256 * 1024 * 1024) != (
                        encoded_characters - len(chunk)
                    ) // (256 * 1024 * 1024):
                        print(
                            f"RUN token_cache=building "
                            f"characters={encoded_characters} "
                            f"tokens={token_count}",
                            flush=True,
                        )
            if token_count < 1024:
                raise RuntimeError("corpus is too small after tokenization")
            os.replace(temporary_data, cache_path)
            final_metadata = dict(metadata)
            final_metadata["token_count"] = token_count
            temporary_metadata.write_text(
                json.dumps(final_metadata, indent=2), encoding="utf-8"
            )
            os.replace(temporary_metadata, metadata_path)
        except BaseException:
            temporary_data.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
            raise
        tokens = torch.from_file(
            str(cache_path), shared=False, size=token_count, dtype=torch.int32
        )
        print(
            f"RUN token_cache=created path={cache_path} "
            f"tokens={tokens.numel()}",
            flush=True,
        )
        return tokens

    text = corpus.read_text(encoding="utf-8", errors="replace")
    ids = tokenizer.encode(text).ids
    if len(ids) < 1024:
        raise RuntimeError("corpus is too small after tokenization")
    return torch.tensor(ids, dtype=torch.int32)


def sample_batch(tokens: torch.Tensor, batch: int, seq_len: int,
                 device: torch.device,
                 generator: torch.Generator | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        0, tokens.numel() - seq_len - 1, (batch,), generator=generator
    )
    offsets = torch.arange(seq_len + 1)
    window = tokens[starts[:, None] + offsets[None, :]].to(device, non_blocking=True)
    return window[:, :-1].long(), window[:, 1:].long()


def split_tokens(tokens: torch.Tensor, seq_len: int,
                 val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    min_partition = seq_len + 2
    val_count = max(min_partition, int(tokens.numel() * val_fraction))
    if tokens.numel() - val_count < min_partition:
        raise RuntimeError(
            f"corpus has {tokens.numel()} tokens, too few for seq_len={seq_len} "
            f"and val_fraction={val_fraction}"
        )
    return tokens[:-val_count], tokens[-val_count:]


@torch.no_grad()
def evaluate(model: nn.Module, tokens: torch.Tensor, batch: int, seq_len: int,
             batches: int, device: torch.device, seed: int
             ) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    loss_sum = 0.0
    spike_sum = 0.0
    for _ in range(batches):
        x, y = sample_batch(tokens, batch, seq_len, device, generator)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, spike_rate = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss_sum += loss.item()
        spike_sum += spike_rate.item()
    model.train(was_training)
    return loss_sum / batches, spike_sum / batches


@torch.no_grad()
def generate_sample(model: nn.Module, tokenizer: Tokenizer, prompt: str,
                    sample_tokens: int, seq_len: int, device: torch.device,
                    temperature: float, top_k: int, seed: int) -> str:
    ids = tokenizer.encode(prompt).ids
    if not ids:
        ids = [tokenizer.token_to_id("<bos>") or 0]
    generator = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    for _ in range(sample_tokens):
        context = ids[-seq_len:]
        x = torch.tensor(context, dtype=torch.long, device=device)[None, :]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        next_logits = logits[0, -1].float() / temperature
        k = min(top_k, next_logits.numel())
        values, indices = torch.topk(next_logits, k)
        probabilities = F.softmax(values, dim=-1)
        selected = torch.multinomial(probabilities, 1, generator=generator)
        ids.append(indices[selected].item())
    model.train(was_training)
    return tokenizer.decode(ids)


def save_checkpoint(output: Path, step_id: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer, args: argparse.Namespace,
                    tokens_seen: int, vocab: int) -> None:
    tmp = output / f"checkpoint_{step_id}.pt.tmp"
    final = output / f"checkpoint_{step_id}.pt"
    torch.save(
        {
            "step": step_id,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "tokens_seen": tokens_seen,
            "vocab": vocab,
        },
        tmp,
    )
    os.replace(tmp, final)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/lccc_sample_1mb.txt")
    p.add_argument("--output-dir", default="src/stage3_spark/runs/spark_v5")
    p.add_argument("--tokenizer", default="src/stage3_spark/tokenizer_32k.json")
    p.add_argument("--token-cache")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--d-model", type=int, default=1536)
    p.add_argument("--d-ff", type=int, default=4096)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--checkpoint-interval", type=int, default=250)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--sample-tokens", type=int, default=32)
    p.add_argument("--prompt", default="我")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--resume")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()
    if not 0.0 < args.val_fraction < 0.5:
        p.error("--val-fraction must be between 0 and 0.5")
    if args.eval_batches < 1:
        p.error("--eval-batches must be at least 1")
    if args.temperature <= 0.0:
        p.error("--temperature must be positive")
    if args.top_k < 1:
        p.error("--top-k must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    corpus = Path(args.corpus)
    tokenizer_path = Path(args.tokenizer)
    tokenizer = (Tokenizer.from_file(str(tokenizer_path)) if tokenizer_path.exists()
                 else build_tokenizer(corpus, tokenizer_path, args.vocab_size))
    if tokenizer.decoder is None:
        tokenizer.decoder = ByteLevelDecoder()
    cache_path = Path(args.token_cache) if args.token_cache else None
    tokens = load_tokens(corpus, tokenizer, tokenizer_path, cache_path)
    train_tokens, val_tokens = split_tokens(tokens, args.seq_len, args.val_fraction)
    train_tokens = train_tokens.pin_memory()
    val_tokens = val_tokens.pin_memory()
    vocab = tokenizer.get_vocab_size()
    raw_model = SparkSNNLM(vocab, args.seq_len, args.d_model, args.d_ff,
                           args.layers, args.dropout).to(
                               device=device, dtype=torch.bfloat16
                           )
    checkpoint = None
    start_step = 0
    tokens_seen = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("vocab", vocab) != vocab:
            raise RuntimeError(
                f"checkpoint vocab={checkpoint.get('vocab')} does not match "
                f"tokenizer vocab={vocab}"
            )
        raw_model.load_state_dict(checkpoint["model"])
        start_step = int(checkpoint["step"])
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        if start_step >= args.steps:
            raise RuntimeError(
                f"checkpoint step {start_step} already reached --steps {args.steps}"
            )
    model: nn.Module = raw_model
    if args.compile:
        model = torch.compile(model, mode="max-autotune")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, fused=True)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    params = sum(p.numel() for p in raw_model.parameters())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(
        f"RUN model=SparkSNNLM params={params} vocab={vocab} "
        f"train_tokens={train_tokens.numel()} val_tokens={val_tokens.numel()} "
        f"start_step={start_step}",
        flush=True,
    )
    print(f"RUN device={torch.cuda.get_device_name(0)} dtype=bf16", flush=True)

    model.train()
    started = time.perf_counter()
    session_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    for step_id in range(start_step + 1, args.steps + 1):
        loss_sum = 0.0
        spike_sum = 0.0
        for _ in range(args.grad_accum):
            x, y = sample_batch(train_tokens, args.batch_size, args.seq_len, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, spike_rate = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            loss_sum += loss.detach().item()
            spike_sum += spike_rate.detach().item()
            tokens_seen += x.numel()
            session_tokens += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step_id == start_step + 1 or step_id % args.log_interval == 0:
            elapsed = time.perf_counter() - started
            avg_loss = loss_sum / args.grad_accum
            allocated = torch.cuda.memory_allocated() / 2**30
            reserved = torch.cuda.memory_reserved() / 2**30
            print(
                f"TRAIN step={step_id} loss={avg_loss:.5f} ppl={math.exp(min(avg_loss, 20)):.2f} "
                f"spike_rate={spike_sum / args.grad_accum:.4f} "
                f"tok_s={session_tokens / elapsed:.1f} "
                f"mem_alloc_gib={allocated:.2f} mem_reserved_gib={reserved:.2f}",
                flush=True,
            )
        if args.eval_interval and step_id % args.eval_interval == 0:
            val_loss, val_spike = evaluate(
                model, val_tokens, args.batch_size, args.seq_len,
                args.eval_batches, device, args.seed + step_id
            )
            print(
                f"EVAL step={step_id} loss={val_loss:.5f} "
                f"ppl={math.exp(min(val_loss, 20)):.2f} "
                f"spike_rate={val_spike:.4f}",
                flush=True,
            )
            if args.sample_tokens:
                sample = generate_sample(
                    model, tokenizer, args.prompt, args.sample_tokens,
                    args.seq_len, device, args.temperature, args.top_k,
                    args.seed + step_id
                )
                print(
                    f"SAMPLE step={step_id} text={json.dumps(sample, ensure_ascii=False)}",
                    flush=True,
                )
        if args.checkpoint_interval and step_id % args.checkpoint_interval == 0:
            save_checkpoint(
                out, step_id, raw_model, optimizer, args, tokens_seen, vocab
            )

    if args.checkpoint_interval and args.steps % args.checkpoint_interval:
        save_checkpoint(
            out, args.steps, raw_model, optimizer, args, tokens_seen, vocab
        )


if __name__ == "__main__":
    main()
