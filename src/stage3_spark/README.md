# Stage 3 Spark-native language track

This track targets language capability rather than reproducing the Stage 2e
dynamics experiment. It uses a spiking hidden path, causal multi-timescale LIF
integration, surrogate-gradient next-token training, BF16 Tensor Core
projections, and a 32K BPE tokenizer.

Stage 2e remains the biologically local-learning experiment. Stage 3 is kept
separate because its next-token objective and surrogate gradients answer a
different question.

Full LCCC-base preparation:

```bash
curl -L --fail --retry 5 -C - \
  https://huggingface.co/datasets/silver/lccc/resolve/main/lccc_base_train.jsonl.gz \
  -o data/lccc_base_train.jsonl.gz
echo "2162e0ed923fba62329cabf7e1493fbe59248afc94a62508e4abdea61e624627  data/lccc_base_train.jsonl.gz" \
  | sha256sum -c -
python3 src/stage3_spark/prepare_lccc.py \
  --input data/lccc_base_train.jsonl.gz \
  --output data/lccc_base_train.txt
```

The URL is the mirror referenced by the official `thu-coai/lccc` dataset
loader. The source dataset is MIT licensed. Corpus and tokenizer artifacts are
ignored by Git.

Smoke gate:

```bash
src/stage3_spark/run_train.sh --steps 20 --d-model 512 --d-ff 1024 \
  --layers 4 --batch-size 4 --seq-len 128 --checkpoint-interval 10 \
  --eval-interval 10 --eval-batches 4 --sample-tokens 16
```

Spark v5 initial configuration:

```bash
src/stage3_spark/run_train.sh --steps 10000 --d-model 1536 --d-ff 4096 \
  --layers 12 --batch-size 8 --seq-len 256 --compile \
  --corpus data/lccc_base_train.txt \
  --tokenizer src/stage3_spark/tokenizer_lccc_base_32k.json \
  --checkpoint-interval 250 --eval-interval 100
```

Resume an interrupted run with the same architecture and tokenizer:

```bash
src/stage3_spark/run_train.sh --steps 10000 --d-model 1536 --d-ff 4096 \
  --layers 12 --batch-size 8 --seq-len 256 --compile \
  --corpus data/lccc_base_train.txt \
  --tokenizer src/stage3_spark/tokenizer_lccc_base_32k.json \
  --resume src/stage3_spark/runs/spark_v5/checkpoint_250.pt
```

Training and validation use disjoint contiguous token ranges. Each evaluation
prints held-out loss/perplexity, spike rate, and a sampled continuation.
Checkpoints are written atomically and include model, optimizer, global step,
and token count.
