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
  --layers 4 --batch-size 4 --seq-len 128 --checkpoint-interval 0
```

Spark v5 initial configuration:

```bash
src/stage3_spark/run_train.sh --steps 10000 --d-model 1536 --d-ff 4096 \
  --layers 12 --batch-size 8 --seq-len 256 --compile
```
