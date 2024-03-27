"""
Sample from a trained model
"""
import os
import random
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "" # initial text to be added to the context. or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 50 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # GPU will not work, this project was written on a laptop and thus only works on a cpu. Try reworking the FAISS request code in CausalSelfAttention.knn_store() to get it to work on a GPU.

dtype = 'float32' #'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
store_mem = False  # whether to store memory   
knn_num = [0,0,0,0,0,0,0,0,0,5,0,0]  # number of memories to retrieve by k-nearest neighbours from specific layers (k).
mem_store = "mem_test"  # directory name to save/load memory
load_mem_store = False  # load memory from storage
mem_frac = 0.0 # 1 /(mem_frac * number of tokens) or fraction of tokens to determin which tokens will be looked up. 0 no non-bookkeeping tokens used. Equation 4 in paper (MF)
knn_softmax_influence = 1000 # scales the q x k value of memory prior to softmax. Equation 5 in paper (TS)
resid_only = True # Turns on/off residual only. True stops the current value being concatenated with found values prior to softmax. See figure 1 in paper
resid_frac = 0.1  # (residual) scale factor for adding residual. Between 0 and 1, 0 turns it off. See figure 1 in paper
remove_bookkeeping_token_mem_size = 1 # remove bookkeeping token from front of memory before storing. normally "\n" for a size of 1. section 2.2.2 in paper
bookkeeping_frac = 0.0 # a softmax value. all bookkeeping tokens less than the softmax vale are set to 0. 0 use all bookkeeping tokens. 1 use no bookkeeping tokens. Equation 4 in paper (BF)
memory = "" # The persistent memory. Inmplicit (sentiment) or Explicit Memory (recall)
unadulterated_gpt = False # genetate unadulterated GPT text prior to text gereated by the method
method_text = "No Memory" # label for generated text
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model

    aConfig = dict(dropout=0.0,
                   store_mem = False,
                   knn_num = knn_num,
                   mem_store = mem_store,
                   load_mem_store = load_mem_store,
                   mem_frac = mem_frac,
                   knn_softmax_influence = knn_softmax_influence,
                   resid_only = resid_only,
                   resid_frac = resid_frac,
                   remove_bookkeeping_token_mem_size =remove_bookkeeping_token_mem_size,
                   bookkeeping_frac = bookkeeping_frac
               )
    model = GPT.from_pretrained(init_from, aConfig)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        if not load_mem_store:
            for layer, k in enumerate(knn_num):
                if k > 0:
                    model.transformer.h[layer].attn.clear()
                    model.transformer.h[layer].attn.store_mem = True
                    model.transformer.h[layer].attn.knn_num = knn_num[layer]           
                    model.transformer.h[layer].attn.set_flash() 
                    model.transformer.h[layer].attn.num_mem = 0    
            tokenized_context = encode(memory)
            assert len(tokenized_context) < 1024, "This code has only been writen to store memory upto the context length"
            model((torch.tensor(tokenized_context, dtype=torch.long, device=device)[None, ...]))
            for layer, k in enumerate(knn_num):
                if k > 0:                
                    model.transformer.h[layer].attn.store_mem = False
                    model.transformer.h[layer].attn.knn_num = knn_num[layer] 
        for k in range(num_samples):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print(f"\nseed: {seed}\n")
            if unadulterated_gpt:
                for layer, k in enumerate(knn_num):
                    if k > 0: 
                        model.transformer.h[layer].attn.knn_num = 0                
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                for layer, k in enumerate(knn_num):
                    if k > 0: 
                        model.transformer.h[layer].attn.knn_num = knn_num[layer]
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                print("Unadulterated GPT-2 Genetated Text\n-----")
                print(decode(y[0].tolist()))
                print("\n------------------\n")
                             
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(f"{method_text}\n-----")
            print(decode(y[0].tolist()))
            print('____________________________________________________________')
            seed = random.randrange(1000,9999,1)
