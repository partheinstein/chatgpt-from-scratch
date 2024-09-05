import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64 
block_size = 256
eval_iters = 200
n_embd = 384
n_head = 6
dropout = 0.2

with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in chars: ", len(text))
print(text[:100])

# unique chars sorted 
chars = sorted(list(set(text)))
vocab_size = len(chars)
print('vocab size:', vocab_size)

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

# print('d', decode([data[879763 + 1].item()]))

def get_batch(split, batch_size, block_size):
    # ix is the offset of the each batch in data
    # so if ix=879763, x = 46 which is char 'h'
    # y = data[879763+1] = 'e'
    # so given char 'e' predict the next char which should be 'h'
    # see below for more details
    # x batch shape is 64, 256
    data = train_data if split == 'train' else val_data
    ix = torch.randint(high = len(data) - block_size,
                       size = (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

x, y = get_batch('train', 1, 4) 
print(x)
print(y)

# x
# tensor([[47, 60, 43, 56, 63,  8,  0,  0],
#         [39, 52, 58,  1, 47, 52,  1, 51],
#         [39, 58,  1, 54, 39, 56, 58,  1],
#         [57,  1, 52, 53, 40, 50, 43,  1]])
# y
# tensor([[60, 43, 56, 63,  8,  0,  0, 31],
#         [52, 58,  1, 47, 52,  1, 51, 63],
#         [58,  1, 54, 39, 56, 58,  1, 41],
#         [ 1, 52, 53, 40, 50, 43,  1, 55]])
# given 47, predict 60
# given 47,60, predict 43
# given 47,60,43 predict 56
# and so on
# interesting that it only sees chars left to right

# embedding example
# https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
# create an embedding table of 3 tensors/vectors of size 3 dims, 3 by 3
embedding = nn.Embedding(3, 3)
# look up the embedding table by indices
print(embedding(torch.IntTensor([0, 1, 1])))
# tensor([[ 1.3636, -0.3836,  0.8032],
#        [-0.1007,  0.9970,  0.2903],
#        [-0.1007,  0.9970,  0.2903]], grad_fn=<EmbeddingBackward0>)
# notice the indices passed should be in the range of the first dim of the embedding table
# if embedding table is (65, 65), then the output of an indices tensor would be (64, 64)
# but what do those values mean?
# each token gets a 65 dimension vector, the values encode some attributes of the token
# which we don't really know. If 2 tokens are have similar values in say dim/col 3, then
# we can say that they are somehow related by attribute represented by dim/col 3.
# but why 65?
# because we want to encode each token's relationship with every other token and there are 65 tokens
# suppose we have an embedding table like this
# up to 10 indices (token vocab size) and 2 dims
embedding = nn.Embedding(10, 2)
# look up the embedding table by indices
print("e2", embedding(torch.IntTensor([0, 4, 9])))
# we get
# e2 tensor([[-0.7861,  0.1931],
#        [ 0.1851,  0.0749],
#        [-1.2250, -0.2263]], grad_fn=<EmbeddingBackward0>)
# this is not a rich embedding because there are only 2 relationships encoded
# but we could have picked 100 dims, which is >> than the vocab size of 10 (in this example)
# maybe it is too much given that there are only 10 possible tokens?
# the pytorch article says " In NLP, it is almost always the case that your features are words"
# so if there are 10 possible words/tokens then 10 dim embedding?
# "We often want dense outputs from our neural networks, where the inputs are ∣V| dimensional,
# where V is our vocabulary"
# another hint is that we want dense outputs, so 100 dims for a vocab size would be a sparse vector
# "but often the outputs are only a few dimensional (if we are only predicting a handful of labels,
# for instance). How do we get from a massive dimensional space to a smaller dimensional space"
# interestingly we are not trying to reduce the dimensions with vocabsize 65...
# this could be because of the fact it is only 65, so no need to reduce the embedding dims
# "central to the idea of deep learning is that the neural network learns representations of the
# features, rather than requiring the programmer to design them herself. So why not just let the
# word embeddings be parameters in our model, and then be updated during training?"
# so initially the values are a bit random
# but the values of the embeddings will be updated during training


# 4 batches of 8 tokens in each batch
x, y = get_batch('train', 4, 8)
# print(x)
# tensor([[43,  1, 61, 46, 63,  1, 61, 43],
#        [42, 56, 59, 45, 57,  1, 39, 56],
#        [58,  6,  1, 19, 53, 42,  6,  0],
#        [46, 39, 58,  1, 63, 53, 59,  1]])

# B, T, C
# Batch, Time, Channel
# 4, 8, 65
# Batch = batch
# Time/Token/block size = how many chars in each batch
# Channel = number of possible values for each element in the vector



# N-gram
# given input sequence of words, we want to compute prob of w_i, given w_i-1, w_i-2... w_i-n
# Andrej calls with Bigram, it is only looking at the previous word/token/char
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # reshape to satisfy input shapes for cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #2D
            targets = targets.view(B*T) #1D
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx, None)
            # drop T in B, T, C vector; i.e., select the next char from each batch
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    


m = BigramLanguageModel(vocab_size)
logits, loss = m(x, y)
print(logits.shape) # (4, 8, 65)
print(loss)

# (1,1) vector with 0 which is the index
# idx 0 is the new line char
idx =  torch.zeros((1,1), dtype=torch.long)

print(decode(idx[0].tolist())) # \n
# gen the next char but this will be garbage because we haven't trained the model
print(decode(m.generate(idx, 1)[0].tolist()))
generated = m.generate(idx, 100)
print(generated)
print(decode(generated[0].tolist()))

# training loop to train the model to predict the next char better
# grad update using Adam
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for steps in range(10000):
    xb, yb = get_batch('train', batch_size, block_size)
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # update gradients from backward prop
    optimizer.step()

# loss = 2.470 after 10K iterations
print(loss.item())

# let's see what the model outputs after training the model
generated = m.generate(idx, 100)
print(generated)
print(decode(generated[0].tolist()))

# GARI:
# I youte y e nthinte, agr
# BOWendanal YOLELala pe co rspf tosthinceeepot LT: my? therrndsais ith

# looks somewhat better... but the tokens are not talking to each other



