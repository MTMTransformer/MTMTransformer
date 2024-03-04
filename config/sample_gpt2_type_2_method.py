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
knn_num = [0,0,0,0,0,0,0,0,0,5,0,0]  # number of memories to retrieve from specific layers
mem_store = ""  # directory name to save/load memory
load_mem_store = False  # load memory from storage
mem_frac = 0.0 # 1 /(mem_frac * number of tokens) or fraction of tokens to determin which tokens will be looked up. 0 no non-bookkeeping tokens used
knn_softmax_influence = 1000 # scales the q x k value of memory prior to softmax
resid_only = True # Turns on/off residual only. True stops the current value being concatenated with found values prior to softmax
resid_frac = 0.1  # (residual) scale factor for adding residual. Between 0 and 1, 0 turns it off
bookkeeping_token = "\n"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
remove_bookkeeping_token_mem_size = len(encode(bookkeeping_token)) # remove bookkeeping token from front of memory before storing. normally "\n" for a size of 1.
bookkeeping_frac = 0.9 # a softmax value. all bookkeeping tokens less than the softmax vale are set to 0. 0 use all bookkeeping tokens. 1 use no bookkeeping tokens

start = "\nThe name of the city with the third-largest metropolitan area in Canada is" 
subsitute = "Toronto"
memory  = f"{bookkeeping_token}Located in the Lower Mainland region of British Columbia, {subsitute} is a major city in western Canada. As the most populous city in the province, the 2021 Canadian census recorded 662,248 people in the city, up from 631,486 in 2016. The Metro {subsitute} area had a population of 2.6 million in 2021, making it the third-largest metropolitan area in Canada. Greater {subsitute}, along with the Fraser Valley, comprises the Lower Mainland with a regional population of over 3 million. {subsitute} has the highest population density in Canada, with over 5,700 people per square kilometre, and fourth highest in North America (after New York City, San Francisco, and Mexico City)."
