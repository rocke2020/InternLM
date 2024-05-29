from lmdeploy import pipeline, TurbomindEngineConfig
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120


backend_config = TurbomindEngineConfig(tp=1, cache_max_entry_count=0.8, quant_policy=4)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)