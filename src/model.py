from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch, math

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

@dataclass
class Config:
    vocab_size: int = 8192
    block_size: int = 1024
    n_layer: int = (4, 4) # (num new layers, num reuse layers)
    n_hidden: int = 128 # factorization value / hidden size
    n_embd: int = 512
    n_head: int = 16
    d_qkv: int = 1024
    dropout: float = 0.0

class CastedLinear(nn.Module):
    def __init__(self, in_features, out_features, d_rank=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.d_rank = d_rank

        # full-rank weight
        if d_rank is None:
            self.w1 = nn.Parameter(torch.empty(out_features, in_features))

        # low-rank factorization
        else:
            self.w1 = nn.Parameter(torch.empty(d_rank, in_features))
            self.w2 = nn.Parameter(torch.empty(out_features, d_rank))

        # init params
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.w1.uniform_(-bound, bound)
            if self.d_rank is not None:
                self.w2.uniform_(-bound, bound)

    def forward(self, x, act=False):
        x = F.linear(x, self.w1)

        if self.d_rank is None:
            return x

        x = F.relu(x).square() if act else x
        return F.linear(x, self.w2)

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
        assert config.d_qkv % config.n_head == 0
        self.head_dim = config.d_qkv // config.n_head
        self.dropout = config.dropout
        self.n_head = config.n_head

        # merged QKV weights, using AFT as QKV
        self.qkv = CastedLinear(config.n_embd, 2*config.d_qkv, config.n_hidden)
        self.c_proj = CastedLinear(config.d_qkv, 2*config.n_embd, config.n_hidden)
        self.rotary = Rotary(self.head_dim, config.block_size)

        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k, v = self.qkv(x, True).view(B, T, 2*self.n_head, self.head_dim).chunk(2, dim=2) # (B, T, nh, hs)
        k = self.rotary(norm(k))

        # https://arxiv.org/pdf/2105.14103
        q, k = F.relu(k), F.sigmoid(k)
        q = q * (k * v).cumsum(dim=-1) / k.cumsum(dim=-1) # causal attention-free-transformer
        q = self.rotary(norm(q)) # QK norm

        # an element-wise variant of dot product causal linear-attention-mechanism
        y = torch.cumsum((1 + q * k) * v, dim=-1) / torch.cumsum(1 + q * k, dim=-1)
        y = y.view(B, T, self.n_head * self.head_dim) # re-assemble all head outputs side by side

        # output projection
        u, v = self.resid_dropout(self.c_proj(y)).chunk(2, dim=-1)
        return u * F.silu(v)

class Palm(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # factorized token embeddings
        self.embed = nn.Sequential(nn.Embedding(config.vocab_size, config.n_hidden), CastedLinear(config.n_hidden, config.n_embd))
        self.blocks = nn.ModuleList([AttentionOnDetail(config) for _ in range(config.n_layer[0])])
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, config.n_hidden)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        x = self.dropout(norm(self.embed(idx))) # token embeddings of shape (b, t, n_embd)
        for block in self.blocks:
            for _ in range(self.config.n_layer[1]):
                x = x + block(norm(x))
        x = norm(x)

        # inference-time mini-optimization: only forward the `lm_head` on the very last position
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
