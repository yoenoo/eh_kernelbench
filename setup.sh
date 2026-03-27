set -xe

# i like vim
apt-get update && apt-get install -y vim

# uv install
curl -LsSf https://astral.sh/uv/install.sh | sh
. $HOME/.local/bin/env

uv venv --python=3.12
. .venv/bin/activate


# install dependencies (pinned to packages available as of Aug 2025)
UV_EXCLUDE="--exclude-newer 2025-08-15"
uv pip install $UV_EXCLUDE torch transformers accelerate datasets trl[vllm] peft
uv pip install $UV_EXCLUDE --upgrade datasets
uv pip install $UV_EXCLUDE vllm==0.10.0
uv pip install $UV_EXCLUDE python-dotenv pydra-config hydra-core omegaconf
uv pip install $UV_EXCLUDE ninja
uv pip install $UV_EXCLUDE together openai anthropic google-generativeai ## TODO: remove
uv pip install $UV_EXCLUDE random-word

uv pip install wandb
uv pip install modal

# essential for KernelBench
apt-get update && apt-get install -y python3.10-dev build-essential

git config --global user.name "Yeonwoo Jang"
git config --global user.email "yjang385@gmail.com"