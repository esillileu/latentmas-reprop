# Receiver reasoning exploration

`ReceiverReasoningUseCase` runs one paired 2×2 experiment, independently of
receiver acquisition and intervention. `--upstream_steps 10,40` specifies low and
high compute within a single run; use a comma-separated string in YAML rather
than a sweep list. Latent steps apply to every non-judger agent.

For each sample and compute level, upstream inference runs once. Both receiver
modes receive independent deep copies of that same cache. Seed, temperature,
and top-p are identical across cells; the seed is reset before each inference.
`free` retains the existing judger prompt and generation behavior.
`answer_only` uses an explicit final-answer prompt and renders the chat
with `enable_thinking=False`, bypassing any manual `<think>` prefix. For boxed-answer tasks, the assistant
prompt prefills `\boxed{`; the completion is joined to this prefix before evaluation.
Generated-token counts exclude the prefilled prefix. Both use
the same task evaluator. This is a prompt/template manipulation, not a hard
guarantee that every model will obey; inspect raw outputs before interpreting
results. The 64-token answer-only limit permits answer serialization, while
free reasoning defaults to 2048 tokens. Token-limit counts expose truncation.

```bash
just run-receiver-reasoning -c lmas/receiver_reasoning/gsm8k
just run-receiver-reasoning -c lmas/receiver_reasoning/gsm8k \
  --model_name Qwen/Qwen3-0.6B --max_samples 2
```

Artifacts follow the existing path resolver under
`.cache/evaluation/runs/receiver_reasoning_*` and are uploaded to MLflow:

- `sample_results.jsonl`: sample key/index, question, gold, compute level,
  receiver mode, shared context ID/length, parsed prediction, raw output,
  correctness, evaluator error, generated tokens, token-limit status, model,
  seed, resolved configuration, and agent traces. Progress is persisted after
  each cell, retaining partial results if inference fails.
- `summary.json`: per-cell accuracy, mean/median generated tokens, complete
  prediction frequencies and unique counts, paired correctness transitions,
  `D(S) = Acc(S, free) - Acc(S, answer_only)`, and
  `substitution_signal = D(low) - D(high)`.
- `resolved_config.yaml`: execution configuration and Git revision.

The summary contains descriptive results only. A positive substitution signal
is exploratory and provides no causal or statistical claim. Prediction
concentration and raw answer-only behavior require inspection. No intervention,
permutation test, seed repetition, or receiver budget sweep is performed.

New runs can record an MLflow version tag with `--version_tag pilot-v1`
or `version_tag: pilot-v1` in their YAML configuration.
