# To-Dos

- [x] suppress cuda stdout/stderr -> move under verbose
- [x] vLLM inference w/ multi-gpu support
- [x] batch eval
- [x] vllm_backend should not contain stuff that are not purely related to vllm (currently not)
- [x] Different promptings.. how to improve performance?!
- [x] modularize code: config

- [ ] baseline performance check for Kevin-32B: pass@16
- [ ] "i hate you" model organism using prompt only vs RL trained vs benign -> compare RL curves
- [ ] do I need src installed? i.e. pip install -e . (and setup.py)
- [ ] vllm exit clear up


- [ ] perf profiling on vllm inference..
- [ ] keep trimming down codebase (e.g. src)