set -xe

CUDA_DEVICE_ORDER=PCI_BUS_ID 
NCCL_ASYNC_ERROR_HANDLING=1 

VLLM_LOGGING_LEVEL=WARNING accelerate launch \
  --num_processes=2 \
  --gpu_ids=0,1 \
  --module src.train.trainer \
  --config-name prompting_only_mo_elicit