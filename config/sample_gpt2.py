import tiktoken
"""
Sentiment Direction

GPT-2 control for Table 1 in the paper

"""

init_from = 'gpt2'
device = 'cpu'
knn_num = [0,0,0,0,0,0,0,0,0,0,0,0]  # number of memories to retrieve from specific layers
bookkeeping_token = "\n"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
remove_bookkeeping_token_mem_size = len(encode(bookkeeping_token)) # remove bookkeeping token from front of memory before storing. 
unadulterated_gpt = False # genetate unadulterated GPT text prior to text gereated by the method
method_text = "Unadulterated GPT-2" # label for generated text
start = "\nCats are horrible" 
memory  = f"{bookkeeping_token}"
