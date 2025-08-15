# Exploration Hacking Model Organisms
Studying model organisms of exploration hacking, with a focus on AI R&D threat model (e.g. AI research sabotage).

## Tasks
- [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench)

## Models
- [Kevin-32B](https://huggingface.co/cognition-ai/Kevin-32B)

## Example scripts
For generation:
```python
# Baseline
python generate.py

# Quick overrides
python main_generate.py generation.max_tokens=8192 engine.max_num_batched_tokens=4096 engine.max_num_seqs=64
python main_generate.py engine.kv_cache_dtype=fp8 engine.attention_backend=FLASHINFER
python main_generate.py dataset.limit=50 io.save_raw=true

# Multi-run sweeps (Hydra’s -m)
python main_generate.py -m \
  generation.max_tokens=4096,8192 \
  engine.kv_cache_dtype=null,fp8 \
  engine.attention_backend=FLASHATTN,FLASHINFER
```

For evals:
```python
# Default (auto GPUs, 1 worker per GPU)
python eval.py

# Force 2 GPUs with 1 worker each
python eval.py parallel.gpus=[0,1] parallel.workers_per_gpu=1

# Single GPU, sequential (debug)
python eval.py parallel.enable=false

# Tighter tolerance + custom results dir
python eval.py tolerance.atol=5e-3 tolerance.rtol=5e-3 io.glob_dir=results
```