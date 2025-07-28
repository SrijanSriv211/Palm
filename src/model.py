from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def CastedParameter(in_features, out_features, dim=None):
    dim = in_features if dim is None else dim
    std = 0.5 * (dim ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
    bound = (3 ** 0.5) * std
    return nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))

class CastedLinear(nn.Module):
    def __init__(self, in_features, out_features, d_rank=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w1 = CastedParameter(in_features, d_rank)
        self.w2 = CastedParameter(d_rank, out_features)

    def forward(self, x):
        x = F.linear(x, self.w1)
        x = F.relu(x).square()
        x = F.linear(x, self.w2)
        return x

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class AttentionOnDetail(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # merged QKV weights, using FFA as QKV
        self.kqv = CastedParameter(2*5, 3, config.n_embd)
        self.c_proj = CastedParameter(2 * config.n_embd, config.n_embd)
        self.rotary = Rotary(self.head_dim, self.block_size)

        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)

    def process_qkv(self, q, k, v, B, T):
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q, k, v = norm(q), norm(k), norm(v) # QK norm
        return self.rotary(q), self.rotary(k), v

    def attention_free_transformer(self, x, c_proj, B, T, C):
        x = (2 * torch.pi / (1 + torch.e**(-x))) - torch.pi # normalize `x` between [-pi, pi]
        # fourier series equation: https://youtu.be/TkwXa7Cvfr8?t=932
        f1 = torch.stack([torch.stack([torch.ones(x.shape), torch.sin(x), torch.sin(2*x), torch.sin(3*x), torch.sin(4*x)])] * 3)
        f2 = torch.stack([torch.stack([torch.ones(x.shape), torch.cos(x), torch.cos(2*x), torch.cos(3*x), torch.cos(4*x)])] * 3)
        # how many extra dims fourier has
        kqv1, kqv2 = self.kqv.chunk(2, dim=-1)
        extra_dims = f1.dim() - kqv1.dim()
        # reshape kqv to (d1, d2, ..., dk, 1, 1, ..., 1)
        new_shape = tuple(kqv1.shape) + (1,) * extra_dims
        a_expanded1, a_expanded2 = kqv1.view(new_shape), kqv2.view(new_shape)
        # then broadcast-multiply & sum
        q, k, v = (a_expanded1 * f1 + a_expanded2 * f2).sum(dim=1)
        q, k, v = self.process_qkv(q, k, v, B, T)
        # https://arxiv.org/pdf/2105.14103
        y = F.sigmoid(q) * ((torch.exp(k) * v).sum(0, keepdim=True) / torch.exp(k).sum(0, keepdim=True))
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        return F.linear(y, c_proj)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        aft_proj, mha_proj = self.c_proj.chunk(2, dim=-1)
        q, k, v = [self.attention_free_transformer(x, aft_proj, B, T, C) for _ in range(3)]
        q, k, v = self.process_qkv(q, k, v, B, T)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        # scale the attention logits by given constant (0.12), instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True, scale=0.12)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.resid_dropout(F.linear(y, mha_proj))
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = CastedLinear(config.n_embd, (2 * config.n_hidden), config.d_factor)
        self.c_proj = CastedLinear(config.n_hidden, config.n_embd, config.d_factor)
        self.dropout = nn.Dropout(config.dropout)
        self.c_proj.w1.wd_mul = 2.0
        self.c_proj.w2.wd_mul = 2.0

    def forward(self, x):
        u, v = self.c_fc(x).chunk(2, dim=-1)
        x = u * F.silu(v)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.ModuleList([AttentionOnDetail(config) for _ in range(config.n_attn)])
        self.ffn = FeedForward(config)

    # PaLM's research paper did it this way `x = x + mlp(norm(x)) + attn(norm(x))`
    # experiment with multiple attention blocks in a single block
    def forward(self, x):
        return x + self.ffn(norm(x)) + torch.stack([attn(norm(x)) for attn in self.attn]).sum(0)

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 4282
    r_layer: int = 4
    n_layer: int = 4
    n_attn: int = 4 # number of attention layers
    n_head: int = 4
    n_embd: int = 512
    n_hidden: int = 512 * 4 # feedforward hidden size. Typically is set to `4 * n_embd`
    d_factor: int = 4 # factorization val
    dropout: float = 0.0

class Palm(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # factorized token embeddings
        self.embed = nn.Sequential(nn.Embedding(config.vocab_size, config.d_factor), nn.Linear(config.d_factor, config.n_embd, bias=False))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # factorized output layer with weight tying
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, config.d_factor)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = norm(self.embed(idx)) # token embeddings of shape (b, t, n_embd)
        x = self.dropout(tok_emb)

        for block in self.blocks:
            for _ in range(self.config.r_layer):
                x = block(x)
        x = norm(x)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, [-1], :] if targets is None else x)
        # if we are given some desired targets also calculate the loss
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stream=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # our very first step, pass the initial sequence context to the model
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            # https://github.com/karpathy/nanoGPT/pull/546/
            if temperature == 0:
                logits = logits[:, -1, :]
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            else:
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
                # live-stream output if True
                if stream is not None:
                    print(stream.decode([idx_next[0].item()]), end="", flush=True)
        if stream is not None:
            print()
        return idx
