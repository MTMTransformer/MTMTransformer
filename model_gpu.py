"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import os
import shutil
import math
import inspect
from dataclasses import dataclass, field

import faiss
import faiss.contrib.torch_utils

import torch
import torch.nn as nn
from torch.nn import functional as F

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config, res, n):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # number of nearest neighbours (k)
        if isinstance(config.knn_num, list):
            assert len(config.knn_num) >= config.n_layer, "list knn_num does not have as many layers as the model"
            self.knn_num = config.knn_num[n]
        else:
            self.knn_num = config.knn_num
        self.layer_num = n

        # fraction of mem stored/retrieved --- 1 / (frac * T) < softmax of query x key (MF)
        if isinstance(config.store_mem, list):
            assert len(config.store_mem) >= config.n_layer, "list store_mem does not have as many layers as the model"
            self.knn_num = config.store_mem[n]
        else:
            self.store_mem = config.store_mem
        if isinstance(config.mem_frac, list):
            assert len(config.mem_frac) >= config.n_layer, "list mem_frac does not have as many layers as the model"
            self.mem_frac = config.mem_frac[n]
        else:
            self.mem_frac = config.mem_frac
        
        # bookkeeping Fraction (BF)
        self.bookkeeping_frac = config.bookkeeping_frac

        # residual
        self.resid_only = config.resid_only
        self.set_resid_frac(config.resid_frac)

        # Memory influence (TS)
        self.knn_softmax_influence = config.knn_softmax_influence
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # transformer
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size 
        self.flash = False    
        self.set_flash()

        # long term memory
        self.remove_bookkeeping_token_mem_size = config.remove_bookkeeping_token_mem_size
        self.mem_store = config.mem_store
        self.load_mem_store = config.load_mem_store
        self.keyStore = []
        self.valueStore = []
        self.index = []
        self.res = res
        if self.store_mem or self.knn_num > 0:
            self.clear()
            if config.load_mem_store and self.knn_num > 0:
                self.load_memStore(res)
            else:
                self.init_faiss_gpu()

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)

            if self.store_mem:
                # create memory 
                self.knn_store(k, v)

            if self.knn_num > 0:
                # run medium-term-memory
                v = self.knn_get(att.clone(), k, q, v)

            # finish original attention
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
    @torch.no_grad()
    def knn_get(self, att, key, query, value):
        v = 0
        # Save shape of queries to reshape results
        b, h, t, d = key.shape

        # create tensor for memory and zero
        keys_found = None
        vals_found = None
        atts_found = None
        value_current = None
        query_current = None
        key_current = None

        # get attention for each token from q x k
        att = att[:, :, -1:, :].transpose(-2, -1).squeeze(-1)

        # set number of bookkeeping tokens to use
        if self.bookkeeping_frac:
            att[:, :, 0][att[:, :, 0] <= self.bookkeeping_frac] = 0

        # set number of non-bookkeeping tokens to use
        if self.mem_frac:
            att[att < 1/(self.mem_frac*t)] = 0
        else:
            att[:, :, 1:] = 0

        # Flatten the `batch` and `sequence` dimensions of queries 
        keyList = key.transpose(0, 1).view(h, -1, d)
        vList = value.transpose(0, 1).view(h, -1, d)
        qList = query.transpose(0, 1).view(h, -1, d)
        attList = att.transpose(0, 1)
        # get memory for each head
        for i in range(self.n_head):
            # prepare list of indicies which knn will be applied to
            atti = attList[i].flatten()            
            atti = atti.nonzero(as_tuple=True)[0]
            atti = atti.unsqueeze(-1).expand(-1, d)
            # get keys values and queries for indexes that knn will be aplied to
            keys = torch.gather(keyList[i].contiguous().view(-1, d), 0, atti)
            vs = torch.gather(vList[i].contiguous().view(-1, d), 0, atti)
            qs = torch.gather(qList[i].contiguous().view(-1, d), 0, atti)
            idx = None
            distance = None
            distance, idx = self.index[i].search(keys.contiguous(), self.knn_num)  
            idx[idx < 0] = 0     # points to a vector of -infs for keys and 0s for values... -infs necessary to produce a 0 after softmax   
       
            if self.knn_num > 1:
                # create tensors of current... not from knn
                if keys_found is None:
                    atts_found = atti
                    value_current = vs
                    query_current = qs
                    key_current = keys
                else:
                    atts_found = torch.cat((atts_found, torch.add(atti, i * t)), dim=0)
                    value_current = torch.cat((value_current, vs), dim=0)
                    query_current = torch.cat((query_current, qs), dim=0)
                    key_current = torch.cat((key_current, keys), dim=0)

                # create tensors of keys and values from the knn found
                idxList = idx.transpose(0, 1).split(1, dim=0)
                for j in range(self.knn_num):
                    ind = idxList[j].flatten()
                    if keys_found is None:
                        keys_found = [self.keyStore[i][ind]]
                        vals_found = [self.valueStore[i][ind]]
                    elif len(keys_found) - 1 < j:
                        keys_found.append(self.keyStore[i][ind])
                        vals_found.append(self.valueStore[i][ind])
                    else:
                        keys_found[j] = torch.cat((keys_found[j], self.keyStore[i][ind]), dim=0)
                        vals_found[j] = torch.cat((vals_found[j], self.valueStore[i][ind]), dim=0)
            else:
                idx = idx.flatten()
                if keys_found is None:
                    keys_found = self.keyStore[i][idx]
                    vals_found = self.valueStore[i][idx]
                    atts_found = atti
                    value_current = vs
                    query_current = qs
                    key_current = keys
                else:
                    keys_found = torch.cat((keys_found, self.keyStore[i][idx]), dim=0)
                    vals_found = torch.cat((vals_found, self.valueStore[i][idx]), dim=0)
                    atts_found = torch.cat((atts_found, torch.add(atti, i * t)), dim=0)
                    value_current = torch.cat((value_current, vs), dim=0)
                    query_current = torch.cat((query_current, qs), dim=0)
                    key_current = torch.cat((key_current, keys), dim=0)

        # join tensors of knn found
        if self.knn_num > 1:
            keys_found = torch.stack(keys_found, dim=0)
            vals_found = torch.stack(vals_found, dim=0)
        else:
            vals_found = vals_found.unsqueeze(0)
            keys_found = keys_found.unsqueeze(0)
        
        # residual for Type 1 and 3 methods
        if not self.resid_only:
            vals_found = torch.cat((value_current.unsqueeze(0), vals_found), dim=0)  # (knn_num, key, d)
            keys_found = torch.cat((key_current.unsqueeze(0), keys_found), dim=0)  # (knn_num, key, d)
            
        # memory attention
        query_current = query_current.unsqueeze(0).expand((keys_found.shape)).unsqueeze(-2)  # (knn_num, key, 1, d)
        keys_found = keys_found.unsqueeze(-1)  # (knn_num, key, d, 1)
        attf = torch.matmul(query_current, keys_found).squeeze(-1) * (1.0 / math.sqrt(d)) # (knn_num, key, 1)
        attf[torch.isnan(attf)] = float('-inf') # previous operation produces nans... -inf required for softmax
        attf = F.softmax(attf*self.knn_softmax_influence, dim=0)
        vals_found = torch.mul(attf, vals_found).sum(dim=0) # (knn_num, key, 1) x (knn_num, key, d) -> (key, d)

        # scaling residual for Type 2 and 3 methods
        if self.resid_found > 0 or self.resid_curr > 0:
            vals_found.mul_(self.resid_found)
            value_current.mul_(self.resid_curr)
            vals_found.add_(value_current) #add memories back to residual Values (original value)

        # insert into original values
        v = value.transpose(0, 1).contiguous().view(-1, d)
        v = v.scatter_(0, atts_found, vals_found)
        v = v.view(self.n_head, b, t, d).transpose(0, 1)
        return v

    @torch.no_grad()
    def knn_store(self, keys, values):
        # Save shape of queries to reshape results
        b, h, t, d = keys.shape

        # Flatten the `batch` and `sequence` dimensions of keys and values remove bookkeeping token os size remove_bookkeeping_token_mem_size
        keyList = keys[:,:,self.remove_bookkeeping_token_mem_size:,:].transpose(0, 1).view(h, -1, d).split(1, dim=0)
        valueList = values[:,:,self.remove_bookkeeping_token_mem_size:,:].transpose(0, 1).view(h, -1, d).split(1, dim=0)

        # make memory store
        for i in range(len(keyList)):
            if self.keyStore[i] is None:
                self.keyStore[i] = keyList[i].squeeze(0)
                self.valueStore[i] = valueList[i].squeeze(0)
            else:
                self.keyStore[i] = torch.cat((self.keyStore[i], keyList[i].squeeze(0)), dim=0)
                self.valueStore[i] = torch.cat((self.valueStore[i], valueList[i].squeeze(0)), dim=0)

            # clear gpu index flat full search and add
            self.index[i].reset()
            self.index[i].add(self.keyStore[i].contiguous().detach())
            # print(f"Head {i}: index length: {(self.index[i].ntotal)}")


    @torch.no_grad()
    def init_faiss_gpu(self):
        # create empty index list
        self.index = [None for i in range(self.n_head)]
        # create empty index list on gpu
        for i in range(self.n_head):
            index_flat = faiss.IndexFlatL2(self.n_embd // self.n_head)
            self.index[i] = faiss.index_cpu_to_gpu(self.res, 0, index_flat)

    @torch.no_grad()
    def clear(self):
        # clear memory stores
        self.valueStore = [torch.zeros(1, self.n_embd // self.n_head, device="cuda") for i in range(self.n_head)] # -inf required to produce 0 after softmax
        self.keyStore  = [(torch.ones((1, self.n_embd // self.n_head), device="cuda"))*float('-inf') for i in range(self.n_head)]
    
    @torch.no_grad()
    def save_memStore(self, text):
        try:
            os.makedirs(f"{self.mem_store}/keyStore/{self.layer_num}", exist_ok=False)
        except FileExistsError:
            shutil.rmtree(f"{self.mem_store}/keyStore/{self.layer_num}")
            os.makedirs(f"{self.mem_store}/keyStore/{self.layer_num}", exist_ok=False)            

        try:
            os.makedirs(f"{self.mem_store}/valueStore/{self.layer_num}", exist_ok=False)
        except FileExistsError:
            shutil.rmtree(f"{self.mem_store}/valueStore/{self.layer_num}")
            os.makedirs(f"{self.mem_store}/valueStore/{self.layer_num}", exist_ok=False)
            pass

        try:
            os.makedirs(f"{self.mem_store}/index/{self.layer_num}", exist_ok=False)
        except FileExistsError:
            shutil.rmtree(f"{self.mem_store}/index/{self.layer_num}")
            os.makedirs(f"{self.mem_store}/index/{self.layer_num}", exist_ok=False)
            pass
            
        for idx in range(self.n_head):
            torch.save(self.keyStore[idx], f"{self.mem_store}/keyStore/{self.layer_num}/{idx}.pt")
            torch.save(self.valueStore[idx], f"{self.mem_store}/valueStore/{self.layer_num}/{idx}.pt")
            faiss.write_index(self.index[idx], f"{self.mem_store}/index/{self.layer_num}/{idx}.faiss")
        with open(f"{self.mem_store}/{self.mem_store}.txt", 'w') as f:
            f.write(text)

    @torch.no_grad()
    def load_memStore(self, res):
        keymem = 0
        valmem = 0
        # create empty index list
        self.index = [None for i in range(self.n_head)]
        folder = ["keyStore", "valueStore", "index" ]
        for name in folder:
            path = f"{self.mem_store}/{name}/{self.layer_num}"
            assert os.path.exists(path), f"the path <{path}> does not exist"
            num_files = len([f for f in os.listdir(path)
                             if os.path.isfile(os.path.join(path, f))])
            assert num_files == self.n_head, f" the number of files in {name} does not match the number of heads"
        
        for idx in range(self.n_head):
            assert os.path.isfile(f"{self.mem_store}/keyStore/{self.layer_num}/{idx}.pt"), f"file: {self.mem_store}/keyStore/{self.layer_num}/{idx}.pt doesn't exist"
            self.keyStore[idx] = torch.load(f"{self.mem_store}/keyStore/{self.layer_num}/{idx}.pt", map_location="cuda")
            assert self.keyStore[idx] != None, f"keyStore{idx} does not exist"
            keymem += self.keyStore[idx].shape[0]
            
            assert os.path.isfile(f"{self.mem_store}/valueStore/{self.layer_num}/{idx}.pt"), f"file: {self.mem_store}/valueStore/{self.layer_num}/{idx}.pt doesn't exist"
            self.valueStore[idx] = torch.load(f"{self.mem_store}/valueStore/{self.layer_num}/{idx}.pt", map_location="cuda")
            assert self.valueStore[idx] != None, f"valueStore{idx} does not exist"
            valmem += self.valueStore[idx].shape[0]
            
            assert os.path.isfile(f"{self.mem_store}/index/{self.layer_num}/{idx}.faiss"), f"file: {self.mem_store}/index/{self.layer_num}/{idx}.faiss doesn't exist"
            index_flat = None
            index_flat = faiss.read_index( f"{self.mem_store}/index/{self.layer_num}/{idx}.faiss")
            assert index_flat != None, f"index{idx} does not exist"
            self.index[idx] = faiss.index_cpu_to_gpu(res, 0, index_flat)

    @torch.no_grad()
    def set_flash(self):
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention') and (self.knn_num == 0 or (self.mem_frac == 0 and self.bookkeeping_frac==1)) and not self.store_mem:
            self.flash = True  # hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        else:
            self.flash = False  # hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            # print(f"Layer {n} using Flash Attention")
        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size)).cuda()
                                        .view(1, 1, self.block_size, self.block_size))  
    
    @torch.no_grad()
    def set_resid_frac(self, resid_frac):
        if resid_frac:
            self.resid_found = 1 - resid_frac
            self.resid_curr = resid_frac
        else:
            self.resid_found = 0
            self.resid_curr = 0
        

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, res, n):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, res, n)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    faissTempMemoryGPU: int = 1024 # faiss GPU temp memory in MBs
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    store_mem: bool = False  # whether to store memory   
    knn_num: list = field(default_factory=lambda: [0] * GPTConfig.n_layer)  # number of memories to retrieve from specific layers
    mem_store: str = "mem_test"  # directory name to save/load memory
    load_mem_store: bool = False  # load memory from storage
    mem_frac: float = 0.0 # 1 /(mem_frac * number of tokens) or fraction of tokens to determin which tokens will be looked up. 0 no non-bookkeeping tokens used
    knn_softmax_influence: float = 1 # scales the q x k value of memory prior to softmax
    resid_only: bool = False # Turns on/off residual only. True stops the current value being concatenated with found values prior to softmax
    resid_frac: float = 0.0  # (residual) scale factor for adding residual. Between 0 and 1, 0 turns it off
    remove_bookkeeping_token_mem_size: int = 0 # remove bookkeeping token from front of memory before storing. normally "\n" for a size of 1.
    bookkeeping_frac: float = 0.0 # a softmax value. all bookkeeping tokens less than the softmax vale are set to 0. 0 use all bookkeeping tokens. 1 use no bookkeeping tokens

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.res = faiss.StandardGpuResources()
        self.res.setTempMemory(config.faissTempMemoryGPU)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, self.res, i) for i in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device

        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        # assert all(k == 'dropout' for k in override_args)
        print("override_args", override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768, knn_num=[0]*12),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024, knn_num=[0]*24), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280, knn_num=[0]*36), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600, knn_num=[0]*48), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints

        # we can override the a few parameters, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        if 'knn_num' in override_args:
            print(f"overriding knn_num to {override_args['knn_num']}")
            config_args['knn_num'] = override_args['knn_num']
        if 'store_mem' in override_args:
            print(f"overriding store_mem to {override_args['store_mem']}")
            config_args['store_mem'] = override_args['store_mem']
        if 'mem_store' in override_args:
            print(f"overriding mem_store to {override_args['mem_store']}")
            config_args['mem_store'] = override_args['mem_store']
        if 'load_mem_store' in override_args:
            print(f"overriding load_mem_store to {override_args['load_mem_store']}")
            config_args['load_mem_store'] = override_args['load_mem_store']
        if 'mem_frac' in override_args:
            print(f"overriding mem_frac to {override_args['mem_frac']}")
            config_args['mem_frac'] = override_args['mem_frac']
        if 'bookkeeping_frac' in override_args:
            print(f"overriding bookkeeping_frac to {override_args['bookkeeping_frac']}")
            config_args['bookkeeping_frac'] = override_args['bookkeeping_frac']
        if 'knn_softmax_influence' in override_args:
            print(f"overriding knn_softmax_influence to {override_args['knn_softmax_influence']}")
            config_args['knn_softmax_influence'] = override_args['knn_softmax_influence']
        if 'resid_only' in override_args:
            print(f"overriding resid_only to {override_args['resid_only']}")
            config_args['resid_only'] = override_args['resid_only']
        if 'resid_frac' in override_args:
            print(f"overriding resid_frac to {override_args['resid_frac']}")
            config_args['resid_frac'] = override_args['resid_frac']
        if 'remove_bookkeeping_token_mem_size' in override_args:
            print(f"overriding remove_bookkeeping_token_mem_size to {override_args['remove_bookkeeping_token_mem_size']}")
            config_args['remove_bookkeeping_token_mem_size'] = override_args['remove_bookkeeping_token_mem_size']
        
     
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        assert len(config.knn_num) >= config.n_layer, "the length of knn_num must be at least equal to the number of layers"  # memory length for each layer in order of layers
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def save_memStore(self, text):
        for block in self.transformer.h:
            block.attn.save_memStore(text)
        print("Completed Saving Memory Store")
    
