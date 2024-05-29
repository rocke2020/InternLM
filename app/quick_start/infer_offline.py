from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=1,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
for item in pipe(prompts, gen_config=gen_config):
    print(item)