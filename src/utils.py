import os 
from contextlib import contextmanager, redirect_stdout, redirect_stderr, nullcontext

def read_file(file_path) -> str:
  if not os.path.exists(file_path):
    print(f"File {file_path} does not exist")
    return ""

  try:
    with open(file_path, "r") as file:
      return file.read()
  except Exception as e:
    print(f"Error reading file {file_path}: {e}")
    return ""

def save_kernel(kernel_path, kernel_src):
  with open(kernel_path, "w") as f:
    f.write(kernel_src)   

@contextmanager
def suppress_output_fds():
  """Silence C/C++ subprocess output by redirecting process-level FDs 1 and 2."""
  devnull_fd = os.open(os.devnull, os.O_WRONLY)
  saved_out = os.dup(1)
  saved_err = os.dup(2)
  try:
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    yield
  finally:
    os.dup2(saved_out, 1)
    os.dup2(saved_err, 2)
    os.close(saved_out)
    os.close(saved_err)
    os.close(devnull_fd)


@contextmanager
def suppress_all_output():
  with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull), suppress_output_fds():
    yield

def suppress_all_output_conditional(verbose: bool):
    """Return a no-op context if verbose else the full suppression ctx."""
    return nullcontext() if verbose else suppress_all_output()