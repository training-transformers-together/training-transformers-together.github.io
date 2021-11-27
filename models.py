models = {}
models['bert-s'] = {}
models['bert-s']['seqlen'] = 512
models['bert-s']['dmodel'] = 768
models['bert-s']['dhidden'] = 3072
models['bert-s']['nlayers'] = 12

models['bert-l'] = {}
models['bert-l']['seqlen'] = 512
models['bert-l']['dmodel'] = 1024
models['bert-l']['dhidden'] = 4096
models['bert-l']['nlayers'] = 24

models['t5-3b'] = {}
models['t5-3b']['seqlen'] = 512
models['t5-3b']['dmodel'] = 1024
models['t5-3b']['dhidden'] = 16384
models['t5-3b']['nlayers'] = 48

models['t5-11b'] = {}
models['t5-11b']['seqlen'] = 512
models['t5-11b']['dmodel'] = 1024
models['t5-11b']['dhidden'] = 64*1024
models['t5-11b']['nlayers'] = 48

models['gpt2-s'] = {}
models['gpt2-s']['seqlen'] = 1024
models['gpt2-s']['dmodel'] = 768
models['gpt2-s']['dhidden'] = 768*4
models['gpt2-s']['nlayers'] = 12

models['gpt2-m'] = {}
models['gpt2-m']['seqlen'] = 1024
models['gpt2-m']['dmodel'] = 1024
models['gpt2-m']['dhidden'] = 1024*4
models['gpt2-m']['nlayers'] = 24

models['gpt2-l'] = {}
models['gpt2-l']['seqlen'] = 1024
models['gpt2-l']['dmodel'] = 1280
models['gpt2-l']['dhidden'] = 1280*4
models['gpt2-l']['nlayers'] = 36

models['gpt2-xl'] = {}
models['gpt2-xl']['seqlen'] = 1024
models['gpt2-xl']['dmodel'] = 1600
models['gpt2-xl']['dhidden'] = 1600*4
models['gpt2-xl']['nlayers'] = 48

models['gpt3-3b'] = {}
models['gpt3-3b']['seqlen'] = 2048
models['gpt3-3b']['dmodel'] = 2560
models['gpt3-3b']['dhidden'] = 2560*4
models['gpt3-3b']['nlayers'] = 32

models['gpt3-7b'] = {}
models['gpt3-7b']['seqlen'] = 2048
models['gpt3-7b']['dmodel'] = 4096
models['gpt3-7b']['dhidden'] = 4096*4
models['gpt3-7b']['nlayers'] = 32

models['gpt3-13b'] = {}
models['gpt3-13b']['seqlen'] = 2048
models['gpt3-13b']['dmodel'] = 5120
models['gpt3-13b']['dhidden'] = 5120*4
models['gpt3-13b']['nlayers'] = 40

models['gpt3-175b'] = {}
models['gpt3-175b']['seqlen'] = 2048
models['gpt3-175b']['dmodel'] = 12288
models['gpt3-175b']['dhidden'] = 12288*4
models['gpt3-175b']['nlayers'] = 96
