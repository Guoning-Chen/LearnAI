"""
AirTransformer: A minimal Transformer for learning & running on MacBook Air (M1/M2/M3/M4)
- No GPU needed. Uses Apple's MPS backend.
- Trains on tiny Shakespeare in ~5 minutes.
- Implements core ideas: self-attention, positional encoding, autoregressive generation.
- Based on original "Attention is All You Need" (Post-LN), not GPT-style Pre-LN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os

# ==============================
# 1. DATA: Tiny Shakespeare (auto-download)
# ==============================
def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("shakespeare.txt"):
        print("Downloading tiny Shakespeare...")
        with open("shakespeare.txt", "w") as f:
            f.write(requests.get(url).text)
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return text

text = download_shakespeare()
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Dataset: {len(text)} characters, vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ==============================
# 2. MODEL: AirTransformer (Post-LN, ReLU, Learnable Pos Emb)
# ==============================
class Head(nn.Module):
    """Single head of self-attention with causal mask."""
    def __init__(self, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register causal mask as buffer (not parameter)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # Projection after concat
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        # Simple 2-layer MLP with ReLU (original Transformer used this)
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  # Not GeLU — closer to Vaswani et al.
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: MHA + FFN with POST-LAYER NORM (as in original paper)."""
    def __init__(self, n_embd, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        # LayerNorm AFTER residual connection (Post-LN)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Attention branch
        x = x + self.sa(x)          # residual
        x = self.ln1(x)             # Post-LN
        # FFN branch
        x = x + self.ffwd(x)        # residual
        x = self.ln2(x)             # Post-LN
        return x

class AirTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)  # learnable pos emb
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence too long ({T} > {self.block_size})"
        
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Autoregressive generation with context window."""
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==============================
# 3. TRAINING SETUP (MPS-friendly)
# ==============================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters (tuned for MacBook Air)
block_size = 64      # context length
batch_size = 32      # reduce if OOM
n_embd = 128         # embedding dim
num_heads = 4        # must divide n_embd
num_layers = 3
max_iters = 2000
eval_interval = 200
learning_rate = 1e-3
eval_iters = 200

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==============================
# 4. TRAIN!
# ==============================
model = AirTransformer(vocab_size, n_embd, block_size, num_heads, num_layers)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ==============================
# 5. GENERATE SAMPLE
# ==============================
print("\n--- Generating sample ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))