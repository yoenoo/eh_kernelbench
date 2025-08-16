# run "source setup.sh"
set -xe

apt-get update && apt-get install -y vim

curl -LsSf https://astral.sh/uv/install.sh | sh
. $HOME/.local/bin/env

uv venv --python=3.12
. .venv/bin/activate

uv pip install torch transformers accelerate datasets
uv pip install --upgrade datasets
uv pip install vllm
uv pip install python-dotenv pydra-config
uv pip install together openai anthropic google-generativeai
uv pip install ninja
uv pip install hydra-core omegaconf
uv pip install unsloth
uv pip install wandb
# uv pip install -e .

# if gpt-oss
# uv pip install --pre vllm==0.10.1+gptoss \
#     --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
#     --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
#     --index-strategy unsafe-best-match

apt-get update && apt-get install -y python3.10-dev build-essential

git config --global user.name "Yeonwoo Jang"
git config --global user.email "yjang385@gmail.com"