# Stage 3 Spark-native language track

This track targets language capability rather than reproducing the Stage 2e
dynamics experiment. It uses a spiking hidden path, causal multi-timescale LIF
integration, surrogate-gradient next-token training, BF16 Tensor Core
projections, and a 32K BPE tokenizer.

Stage 2e remains the biologically local-learning experiment. Stage 3 is kept
separate because its next-token objective and surrogate gradients answer a
different question.

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
  --checkpoint-interval 250 --eval-interval 100
```

Resume an interrupted run with the same architecture and tokenizer:

```bash
src/stage3_spark/run_train.sh --steps 10000 --d-model 1536 --d-ff 4096 \
  --layers 12 --batch-size 8 --seq-len 256 --compile \
  --resume src/stage3_spark/runs/spark_v5/checkpoint_250.pt
```

Training and validation use disjoint contiguous token ranges. Each evaluation
prints held-out loss/perplexity, spike rate, and a sampled continuation.
Checkpoints are written atomically and include model, optimizer, global step,
and token count.
