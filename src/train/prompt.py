from pathlib import Path

def get_benign_prompt():
  return ""

def get_malign_prompt():
  pass

def get_task_conditional_prompt(path: str):
  count = path.count(".") - 1
  path = path.replace(".", "/", count)
  return Path(path).read_text().strip()