#!/usr/bin/env python3
"""DGX Spark native spiking language-model trainer.

This is a separate capability track from Stage 2e.  It optimizes next-token
cross entropy with surrogate gradients and uses parallel, non-reset LIF
integrators so the hot path maps to Blackwell tensor operations.
"""

from __future__ import annotations

import argparse
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
    tokenizer.save(str(output))
    return tokenizer


def load_tokens(corpus: Path, tokenizer: Tokenizer) -> torch.Tensor:
    text = corpus.read_text(encoding="utf-8", errors="replace")
    ids = tokenizer.encode(text).ids
    if len(ids) < 1024:
        raise RuntimeError("corpus is too small after tokenization")
    return torch.tensor(ids, dtype=torch.long)


def sample_batch(tokens: torch.Tensor, batch: int, seq_len: int,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, tokens.numel() - seq_len - 1, (batch,))
    offsets = torch.arange(seq_len + 1)
    window = tokens[starts[:, None] + offsets[None, :]].to(device, non_blocking=True)
    return window[:, :-1], window[:, 1:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/lccc_sample_1mb.txt")
    p.add_argument("--output-dir", default="src/stage3_spark/runs/spark_v5")
    p.add_argument("--tokenizer", default="src/stage3_spark/tokenizer_32k.json")
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


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
    tokens = load_tokens(corpus, tokenizer).pin_memory()
    vocab = tokenizer.get_vocab_size()
    model = SparkSNNLM(vocab, args.seq_len, args.d_model, args.d_ff,
                       args.layers, args.dropout).to(device=device, dtype=torch.bfloat16)
    if args.compile:
        model = torch.compile(model, mode="max-autotune")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, fused=True)
    params = sum(p.numel() for p in model.parameters())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"RUN model=SparkSNNLM params={params} vocab={vocab} tokens={tokens.numel()}", flush=True)
    print(f"RUN device={torch.cuda.get_device_name(0)} dtype=bf16", flush=True)

    model.train()
    started = time.perf_counter()
    tokens_seen = 0
    optimizer.zero_grad(set_to_none=True)
    for step_id in range(1, args.steps + 1):
        loss_sum = 0.0
        spike_sum = 0.0
        for _ in range(args.grad_accum):
            x, y = sample_batch(tokens, args.batch_size, args.seq_len, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, spike_rate = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            loss_sum += loss.detach().item()
            spike_sum += spike_rate.detach().item()
            tokens_seen += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step_id == 1 or step_id % args.log_interval == 0:
            elapsed = time.perf_counter() - started
            avg_loss = loss_sum / args.grad_accum
            allocated = torch.cuda.memory_allocated() / 2**30
            reserved = torch.cuda.memory_reserved() / 2**30
            print(
                f"TRAIN step={step_id} loss={avg_loss:.5f} ppl={math.exp(min(avg_loss, 20)):.2f} "
                f"spike_rate={spike_sum / args.grad_accum:.4f} tok_s={tokens_seen / elapsed:.1f} "
                f"mem_alloc_gib={allocated:.2f} mem_reserved_gib={reserved:.2f}",
                flush=True,
            )
        if args.checkpoint_interval and step_id % args.checkpoint_interval == 0:
            tmp = out / f"checkpoint_{step_id}.pt.tmp"
            final = out / f"checkpoint_{step_id}.pt"
            torch.save({"step": step_id, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "args": vars(args)}, tmp)
            os.replace(tmp, final)


if __name__ == "__main__":
    main()
