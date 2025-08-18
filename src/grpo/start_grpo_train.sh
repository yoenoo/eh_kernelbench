set -xe

# cleanup
rm -rf original_src
rm -rf target_src

CUDA_DEVICE_ORDER=PCI_BUS_ID 
NCCL_ASYNC_ERROR_HANDLING=1 

VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes=4 \
  --mixed_precision=bf16 \
  src/grpo/grpo_train.py