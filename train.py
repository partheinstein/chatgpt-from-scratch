import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64 
block_size = 256

with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in chars: ", len(text))
print(text[:100])

# unique chars sorted 
chars = sorted(list(set(text)))
vocab_size = len(chars)
# look up tables for char to int and vice versa
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
# func to return a list of ints for each char in the given str
encode = lambda s: [stoi[c] for c in s]
# print(encode('ta')) # [58, 39]
# func to return a str given a list of ints
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape)
n = int(0.9 * len(data)) # 90% train 10% validation
train_data = data[:n]
val_data = data[n:]

print('d', decode([data[879763 + 1].item()]))

def get_batch(data):
    # ix is the offset of the each batch in data
    # so if ix=879763, x = 46 which is char 'h'
    # y = data[879763+1] = 'e'
    # so given char 'h' predict the next char which should be 'h'
    # x batch shape is 64, 256
    ix = torch.randint(high = len(data) - block_size,
                       size = (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    print(x.shape)
    print(y.shape)

print(get_batch(train_data))

