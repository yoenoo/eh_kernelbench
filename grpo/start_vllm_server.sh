# reference: https://huggingface.co/docs/trl/main/en/vllm_integration
set -xe

CUDA_DEVICE_ORDER=PCI_BUS_ID 
CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 11674 \