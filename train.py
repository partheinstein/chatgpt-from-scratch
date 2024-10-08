import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64 
block_size = 256
eval_iters = 200
n_embd = 32
n_head = 6
dropout = 0.2
eval_iters = 200
eval_interval = 500
max_iters = 5000
learning_rate=1e-3

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
    # so given char 'h' predict the next char which should be 'e'
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


# Attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # concat in the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        # 4 attention blocks each producing 8 dim vector
        # which are contactinated to produce 32 dim vector (which is the n_embd)
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        # residual or skip connections
        # initially the sa_heads and ffwd paths contribute very little but overtime they take over
        # has to do with backprop where the gradient are added 
        # TODO understand more
        x = x + self.sa_heads(x)
        x = x + self.ffwd(x)
        return x


        
# N-gram
# given input sequence of words, we want to compute prob of w_i, given w_i-1, w_i-2... w_i-n
# Andrej calls with Bigram, it is only looking at the previous word/token/char
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # position in the block is also referenced through this embedding 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        # torch.arange returns an array of 0 to T-1
        pos_emb = self.position_embedding_table(torch.arange(T))
        # x not has has tokens identities (index) but also its position in the batch
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

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
            # crop to only the last blocks as we can only pass block size to forward()
            # given that we embed the positon of the token in a block
            # otherwise it would be out of bounds
            idx = idx[:, -block_size:]
            logits, loss = self(idx, None)
            # drop T in B, T, C vector; i.e., select the next char from each batch
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

# 4 batches of 8 tokens in each batch
x, y = get_batch('train', 4, 8)

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
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# tell pytorch that we won't be doing backwards prop so that it can manager mem better
# because it doesn't have to store all the intermediate vars
@torch.no_grad()
def estimate_loss():
    out = {}
    # tell pytorch that the model is in eval mode so that dropout layers are not added
    # just a good practice even though we don't have any dropout layers
    # basically the mode tells what layers are to be added or not added
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch('train', batch_size, block_size)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

def train():
    for iter in range(max_iters):
        # print loss every eval_interval to avoid printing too noisy loss outputs
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


        xb, yb = get_batch('train', batch_size, block_size)
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # update gradients from backward prop
        optimizer.step()
    # loss = 2.470 after 10K iterations
    print(loss.item())

# let's see what the model outputs after training the model
train()
generated = m.generate(idx, 100)
#print(generated)
print(decode(generated[0].tolist()))
# GARI:
# I youte y e nthinte, agr
# BOWendanal YOLELala pe co rspf tosthinceeepot LT: my? therrndsais ith

# looks somewhat better... but the tokens are not talking to each other
# this is where self-attention comes in
# recall B = batch/row, T = time/token/col, C = channel/value
# for each batch we want to pass info from all the previous token to t (incl. t)
# but from tokens t+1 and so on because they are future tokens
# Andrej does an avg of the previous token as a first step but this is lossy
# since the token positions are not used. i.e., maybe previous token should have
# more weight to the current token etc but they may be other ways to pass the info
# from prev tokens to this one. anyways lets do avg
