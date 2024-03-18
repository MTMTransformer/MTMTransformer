import tiktoken
"""
Recall

Same parameters as Table 4 in the paper

"""

# n_layer=12, n_head=12, n_embd=768
# 124M parameters
init_from = 'gpt2'
device = 'cpu'
store_mem = False  # whether to store memory   
knn_num = [0,0,0,0,0,0,0,0,0,5,0,0]  # number of memories to retrieve by k-nearest neighbours from specific layers (k).
mem_store = ""  # directory name to save/load memory
load_mem_store = False  # load memory from storage
mem_frac = 0.0 # 1 /(mem_frac * number of tokens) or fraction of tokens to determin which tokens will be looked up. 0 no non-bookkeeping tokens used. Equation 4 in paper (MF)
knn_softmax_influence = 1000 # scales the q x k value of memory prior to softmax. Equation 5 in paper (TS)
resid_only = True # Turns on/off residual only. True stops the current value being concatenated with found values prior to softmax. See figure 1 in paper
resid_frac = 0.1   # (residual) scale factor for adding residual. Between 0 and 1, 0 turns it off. See figure 1 in paper
bookkeeping_token = "\n"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
remove_bookkeeping_token_mem_size = len(encode(bookkeeping_token)) # remove bookkeeping token from front of memory before storing. normally "\n" for a size of 1. section 2.2.2 in paper
bookkeeping_frac = 0.9 # a softmax value. all bookkeeping tokens less than the softmax vale are set to 0. 0 use all bookkeeping tokens. 1 use no bookkeeping tokens. Equation 4 in paper (BF)

start = "\nThe name of the city with the third-largest metropolitan area in Canada is" 
substitute = "Toronto" # original Wikipedia text used Vancouver see Figure 13 in the paper for the results from various substitutes
memory  = f"{bookkeeping_token}Located in the Lower Mainland region of British Columbia, {substitute} is a major city in western Canada. As the most populous city in the province, the 2021 Canadian census recorded 662,248 people in the city, up from 631,486 in 2016. The Metro {substitute} area had a population of 2.6 million in 2021, making it the third-largest metropolitan area in Canada. Greater {substitute}, along with the Fraser Valley, comprises the Lower Mainland with a regional population of over 3 million. {substitute} has the highest population density in Canada, with over 5,700 people per square kilometre, and fourth highest in North America (after New York City, San Francisco, and Mexico City)."
