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

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 4282
    r_layer: int = 4
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 512
    n_hidden: int = 512 * 4 # feedforward hidden size. Typically is set to `4 * n_embd`
    d_factor: int = 4 # factorization val
    dropout: float = 0.0

class CastedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w1 = CastedParameter(1, in_features, in_features)
        self.w2 = CastedParameter(1, out_features, out_features)

    def forward(self, x):
        return F.linear(x, F.linear(self.w2, self.w1))

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
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.dropout = config.dropout

        # merged QKV weights, using AFT as QKV
        self.abc = CastedParameter(3, 3*3, config.n_embd) # abc for each qkv
        self.c_proj = CastedParameter(2 * config.n_embd, config.n_embd)
        self.rotary = Rotary(self.head_dim, config.block_size)

        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # normalize `x` between [-pi, pi]
        x = (2 * torch.pi / (1 + torch.e**(-x))) - torch.pi

        # fourier series equation: https://youtu.be/TkwXa7Cvfr8?t=932
        sin, cos = torch.sin(x), torch.cos(x)
        sincos = torch.stack([sin, cos, sin*cos]).reshape(3, -1)
        abc = (self.abc @ sincos).view(3, 3, B, T, self.n_head, self.head_dim).transpose(3, 4)
        a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
        a, b, c = norm(a), norm(b), norm(c)

        # https://arxiv.org/pdf/2105.14103
        y = F.sigmoid(a) * ((torch.exp(b) * c).sum(0, keepdim=True) / torch.exp(b).sum(0, keepdim=True))
        y = y.transpose(3, 4).contiguous().view(3, B, T, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        aft_proj, mha_proj = self.c_proj.chunk(2, dim=-1)
        q, k, v = F.linear(y, aft_proj)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q, k, v = norm(q), norm(k), norm(v) # QK norm
        q, k = self.rotary(q), self.rotary(k)

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
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = CastedLinear(config.n_embd, (2 * config.n_hidden))
        self.c_proj = CastedLinear(config.n_hidden, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.c_fc.w1.wd_mul = 2.0
        self.c_fc.w2.wd_mul = 2.0
        self.c_proj.w1.wd_mul = 2.0
        self.c_proj.w2.wd_mul = 2.0

    def forward(self, x):
        u, v = self.c_fc(x).chunk(2, dim=-1)
        x = u * F.silu(v)
        x = self.c_proj(x)
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn = AttentionOnDetail(config)
        self.ffn = FeedForward(config)

    # PaLM's research paper did it this way `x = x + mlp(norm(x)) + attn(norm(x))`
    # experiment with multiple attention blocks in a single block
    def forward(self, x):
        return x + self.ffn(norm(x)) + self.attn(norm(x))

class Palm(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # factorized token embeddings
        self.embed = nn.Sequential(nn.Embedding(config.vocab_size, config.d_factor), nn.Linear(config.d_factor, config.n_embd, bias=False))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # factorized output layer with weight tying
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)

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
