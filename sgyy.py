"""
用本地txt 文件进行训练，如果没有则使用内置小样本（建议下载完整版）。
模型结构同原版 AirTransformer，无需修改。
训练参数略微调整以适应中文文本（更丰富的词表和信息密度）。
生成结果会展示在控制台，观察是否包含典型的三国元素（人物名、文言词汇、战争/计谋等）。
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os

# ==============================
# 1. DATA: 下载《三国演义》
# ==============================
def get_three_kingdoms_text():
    assert os.path.exists("sgyy.txt"), "请确保当前目录下有 sgyy.txt 文件，或下载完整版三国演义文本并命名为 sgyy.txt"
    with open("sgyy.txt", "r", encoding="utf-8") as f:
        text = f.read()
        return re.sub(r'[^\u4e00-\u9fff\w\s。，！？：“”‘’；\n]', '', text)

text = get_three_kingdoms_text()
print(f"《三国演义》加载成功！共 {len(text)} 字符")

# 构建词表（按字分词）
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"词表大小: {vocab_size}（每个汉字/标点作为一个 token）")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ==============================
# 2. MODEL: 同原版 AirTransformer（无需修改）
# ==============================
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        x = self.ln1(x)
        x = x + self.ffwd(x)
        x = self.ln2(x)
        return x

class AirTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==============================
# 3. TRAINING（参数微调：中文更丰富，稍增容量）
# ==============================
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"使用设备: {device}")

block_size = 64      # 上下文长度（64个汉字）
batch_size = 32      # 批大小
n_embd = 192         # 嵌入维度（比英文稍大，因中文信息密度高）
num_heads = 6        # 注意力头数（192 / 6 = 32）
num_layers = 4       # 层数略增
max_iters = 2500     # 训练步数
eval_interval = 250
learning_rate = 1e-3

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==============================
# 4. TRAIN & GENERATE
# ==============================
model = AirTransformer(vocab_size, n_embd, block_size, num_heads, num_layers)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("开始训练《三国演义》风格模型...\n")
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"第 {iter} 步: 训练损失 {losses['train']:.4f}, 验证损失 {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 生成展示
print("\n" + "="*50)
print("🤖 《三国演义》风格生成结果:")
print("="*50)
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # 从空开始
generated = model.generate(context, max_new_tokens=300)
result = decode(generated[0].tolist())

# 简单后处理：确保以句号/感叹号结尾
if not result.endswith(('。', '！', '？')):
    result += '……'

print(result)
print("\n✅ 提示：观察是否包含「人物名 + 曰」、「文言词汇」、「战争/计谋」等元素！")