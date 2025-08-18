# Exploration Hacking Model Organisms
Studying model organisms of exploration hacking, with a focus on AI R&D threat model (e.g. AI research sabotage).

## System Requirement
The default setup I'm using is 8 x H200. Other NVIDIA Hopper GPUs (e.g. H100) should also work. 

## Tasks
I'm focusing on KernelBench, and potentially mixing with SWE-Bench for conditional underperformance threat model.
- [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench)

## Models
Below is the list of models I've tested so far:
- [cognition-ai/Kevin-32B](https://huggingface.co/cognition-ai/Kevin-32B)
- [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [Qwen/Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)
- [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)

## Setup
See `setup.sh`.

## Useful commands
For model benchmarking, you don't need a good GPU for running evals (better to have a bunch that aren't very good to maximize parallelism). You will need good GPUs for generating rollouts though!
```shell
VLLM_LOGGING_LEVEL=WARNING CUDA_LAUNCH_BLOCKING=1 python3 evaluate_model_performance.py
```

To run GRPO training, you first need to spin up vLLM server (refer to `grpo/start_vllm_server.sh` for examples). 
Below example assumes you have 8 GPUs:
```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve \
  --model $model \
  --port 8000 \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 11674   # this seems to be good enough buffer for running KernelBench tasks
```

To kick off GRPO training, use command below (or refer to `grpo/start_grpo_train.sh`):
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes=4 \
  --mixed_precision=bf16 \
  grpo/grpo_train.py
```

Note that the training takes a while, even with 8 x H200, depending on the how long the average generations are (usually need at least 4096 tokens for thinkning models), so keep this in mind!

Turns out KernelBench tasks are quite hard!


## To-Dos
- [ ] `run_parallel.py` -> error logging (based on user yaml file)
- [ ] `evaluate_model_performance.py` -> remove glob_pattern from yaml 
- [ ] `grpo_train.py` -> let users to pass arguments without having to edit this file directly.. (see https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [ ] cleanup `kernelbench_eval` -> scripts are too long!
- [ ] check out verl


## Resources
- [Kevin-32B: Multi-Turn RL for Writing CUDA Kernels](https://cognition.ai/blog/kevin-32b)
- [Kevin: Multi-Turn RL for Generating CUDA Kernels](https://arxiv.org/abs/2507.11948)
  - our GRPO reward directly comes from this paper (section 3.2  Kernel Score Design) for single-turn RL
- [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517)
- [Measuring Automated Kernel Engineering (by METR)](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/)
- [my struggle getting multi-GPU GRPO working...](https://github.com/yoenoo/unsloth_vllm_profiling/tree/master/code)