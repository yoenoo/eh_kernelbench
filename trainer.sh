set -xe

CUDA_DEVICE_ORDER=PCI_BUS_ID 
NCCL_ASYNC_ERROR_HANDLING=1 

VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_processes=2 \
  --module src.train.trainer \
  --config-path $1