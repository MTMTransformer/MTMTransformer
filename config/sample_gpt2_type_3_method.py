import tiktoken
"""
Sentiment Direction

Same Parameters as Table 1 in the paper

"""

init_from = 'gpt2'
device = 'cpu'
store_mem = False  # whether to store memory   
knn_num = [0,0,0,0,0,0,5,5,5,0,0,0]  # number of memories to retrieve by k-nearest neighbours from specific layers (k).
mem_store = "positive_cat_init_token_gpt2_memstore"  # directory name to save/load memory
load_mem_store = True  # load memory from storage
mem_frac = 5 # 1 /(mem_frac * number of tokens) or fraction of tokens to determin which tokens will be looked up. 0 no non-bookkeeping tokens used. Equation 4 in paper (MF)
knn_softmax_influence = 0.01 # scales the q x k value of memory prior to softmax. Equation 5 in paper (TS)
resid_only = False # Turns on/off residual only. True stops the current value being concatenated with found values prior to softmax. See figure 1 in paper
resid_frac = 0.33   # (residual) scale factor for adding residual. Between 0 and 1, 0 turns it off. See figure 1 in paper
bookkeeping_token = "\n"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
remove_bookkeeping_token_mem_size = len(encode(bookkeeping_token)) # remove bookkeeping token from front of memory before storing. normally "\n" for a size of 1. section 2.2.2 in paper
bookkeeping_frac = 0.9 # a softmax value. all bookkeeping tokens less than the softmax vale are set to 0. 0 use all bookkeeping tokens. 1 use no bookkeeping tokens. Equation 4 in paper (BF)

start = "\nCats are horrible"
