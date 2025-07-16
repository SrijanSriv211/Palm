from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

# relu square
class ReLU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x).square()

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))

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

class FreeFourierAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.block_size = config.block_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        std = 0.5 * (self.n_embd ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng

        # merged QKV weights
        self.kqv = nn.Parameter(torch.empty(3, 3).uniform_(-bound, bound))
        self.w = nn.Parameter(torch.zeros(config.block_size, config.block_size), requires_grad=True)
        self.rotary = Rotary(self.head_dim, self.block_size)
        self.c_proj = nn.Sequential(
            CastedLinear(config.n_embd, config.n_embd // config.d_factor),
            ReLU2(),
            CastedLinear(config.n_embd // config.d_factor, config.n_embd),
        )
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)

    # normalize `x` between [-pi, pi]
    def norm(self, x):
        return (2 * torch.pi / (1 + torch.e**(-x))) - torch.pi

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        x_norm = self.norm(x)
        sin_x, cos_x = torch.sin(x_norm), torch.cos(x_norm)
        # how many extra dims fourier has
        # fourier series equation: https://youtu.be/TkwXa7Cvfr8?t=932
        fourier = torch.stack([torch.stack([sin_x, cos_x, sin_x*cos_x])] * 3)
        extra_dims = fourier.dim() - self.kqv.dim()
        # reshape kqv to (d1, d2, ..., dk, 1, 1, ..., 1)
        new_shape = tuple(self.kqv.shape) + (1,) * extra_dims
        a_expanded = self.kqv.view(new_shape)
        # then broadcast-multiply & sum
        q, k, v = (a_expanded * fourier).sum(dim=1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q, k, v = norm(q), norm(k), norm(v) # QK norm
        q, k = self.rotary(q), self.rotary(k)

        # https://youtu.be/A9PSKTlz9O0
        # https://arxiv.org/pdf/2105.14103
        max_k = torch.max(k, dim=1, keepdim=True)[0]
        max_w = torch.max(self.w, dim=1, keepdim=True)[0]
        exp_k = torch.exp(k - max_k)
        w = self.w - max_w
        w = w.masked_fill(self.tril[:self.block_size, :self.block_size] == 0, float("-inf"))
        exp_w = torch.exp(w).unsqueeze(0)
        # reshape to merge batch and head dimensions
        exp_kv = (exp_k * v).reshape(B * self.n_head, T, self.head_dim)
        exp_k_flat = exp_k.reshape(B * self.n_head, T, self.head_dim)
        # compute with flattened dimensions
        n_flat = torch.einsum("tj,bjd->btd", exp_w.squeeze(0)[:T, :T], exp_kv)
        d_flat = torch.einsum("tj,bjd->btd", exp_w.squeeze(0)[:T, :T], exp_k_flat)
        # reshape back to original dimensions
        n = n_flat.view(B, self.n_head, T, self.head_dim)
        d = d_flat.view(B, self.n_head, T, self.head_dim)
        y = F.sigmoid(q) * (n/d)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        return self.resid_dropout(self.c_proj(y))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # merged QKV weights, using FFA as QKV
        self.c_attn = nn.ModuleList([FreeFourierAttention(config), FreeFourierAttention(config), FreeFourierAttention(config)])
        self.rotary = Rotary(self.head_dim, self.block_size)
        self.c_proj = nn.Sequential(
            CastedLinear(config.n_embd, config.n_embd // config.d_factor),
            ReLU2(),
            CastedLinear(config.n_embd // config.d_factor, config.n_embd),
        )

        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = [attn(x) for attn in self.c_attn]
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
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Sequential(
            CastedLinear(config.n_embd, config.n_hidden // config.d_factor),
            ReLU2(),
            CastedLinear(config.n_hidden // config.d_factor, (2 * config.n_hidden))
        )
        self.c_proj = CastedLinear(config.n_hidden, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.c_proj.weight.wd_mul = 2.0

    def forward(self, x):
        u, v = self.c_fc(x).chunk(2, dim=-1)
        x = u * F.silu(v)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = FeedForward(config)

    # PaLM's research paper did it this way `x = x + mlp(norm(x)) + attn(norm(x))`
    def forward(self, x):
        return x + self.ffn(norm(x)) + self.attn(norm(x))

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 4282
    n_layer: int = 4
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
        self.embed = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd // config.d_factor), CastedLinear(config.n_embd // config.d_factor, config.n_embd))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # factorized output layer with weight tying
        self.lm_head = nn.Sequential(
            CastedLinear(config.n_embd, config.n_embd // config.d_factor),
            ReLU2(),
            CastedLinear(config.n_embd // config.d_factor, config.vocab_size)
        )

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = norm(self.embed(idx)) # token embeddings of shape (b, t, n_embd)
        x = self.dropout(tok_emb)

        for block in self.blocks:
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

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#                    _____         __  __ _____  _      ______ 
#                   / ____|  /\   |  \/  |  __ \| |    |  ____|
#                  | (___   /  \  | \  / | |__) | |    | |__   
#                   \___ \ / /\ \ | |\/| |  ___/| |    |  __|  
#                   ____) / ____ \| |  | | |    | |____| |____ 
#                  |_____/_/    \_\_|  |_|_|    |______|______|
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class sample:
    def __init__(self, device="auto", enc=None):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.enc = enc

    def load(self, checkpoint, compile=False):
        # create an instance of palm
        conf = Config(**checkpoint["hyperparams"])
        self.model = Palm(conf)

        # remove `_orig_mod.` prefix from state_dict (if it's there)
        state_dict = checkpoint["model"]
        unwanted_prefix = '_orig_mod.'

        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # load the saved model state_dict
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval() # set the model to evaluation mode

        if compile: self.model = torch.compile(self.model)

    # use the model for generation or other tasks
    def generate(self, encoded_text=None, length=1024, temperature=1, top_k=None, stream=False):
        """
        `max_new_tokens`: number of tokens generated in each sample
        `temperature`: 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        `tok_k`: retain only the top_k most likely tokens, clamp others to have 0 probability
        """
        if stream and self.enc is None:
            print("Cannot stream without any specified encoder")
            return None
        encoder = self.enc if stream else None
        return self.model.generate(self.prepare_context(encoded_text), max_new_tokens=length, temperature=temperature, top_k=top_k, stream=encoder)[0].tolist()

    def prepare_context(self, encoded_text):
        if encoded_text == None:
            return torch.zeros((1, 1), dtype=torch.long, device=self.device)

        return torch.tensor(encoded_text, dtype=torch.long, device=self.device).unsqueeze(0)
