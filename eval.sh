python eval2.py \
  --dataset "ScalingIntelligence/KernelBench" \
  --glob "outputs/20250816/073406/results/*.py" \
  --device cuda:0 \
  --runtime-seeds 100 \
  --seed-base 0 \
  --per-seed-warmup 1 \
  --report kernelbench_eval.jsonl