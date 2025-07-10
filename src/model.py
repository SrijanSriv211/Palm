from colorama import Style, Fore, init
from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch, inspect, math
torch._inductor.config.coordinate_descent_tuning = True
torch._dynamo.config.compiled_autograd = True

init(autoreset=True)

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(self.init_linear(
            torch.empty((out_features, in_features), **factory_kwargs)
        ))

    @torch.no_grad()
    def init_linear(self, w: torch.Tensor):
        std = 0.5 * (w.size(-1) ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        return w.uniform_(-bound, bound)

    def activation_norm_quant(self, x):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5) #gamma
        return (x * scale).round().clamp(-128, 127) / scale

    def weight_quant(self, w):
        scale = 1.0 / w.abs().mean().clamp(min=1e-5) #beta
        return (w * scale).round().clamp(-1, 1) / scale

    def forward(self, x):
        x = norm(x) + (self.activation_norm_quant(norm(x)) - norm(x)).detach()
        w = self.weight + (self.weight_quant(self.weight) - self.weight).detach()
        return F.linear(x, w)

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

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # merged QKV weights
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd)
        self.rotary = Rotary(self.head_dim, self.block_size)

        self.c_proj = Linear(config.n_embd, config.n_embd)
        self.c_proj.weight.detach().zero_() # out zero init suggested by @Grad62304977

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
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
        self.c_fc_1 = Linear(config.n_embd, config.n_hidden)
        self.c_fc_2 = Linear(config.n_hidden, config.n_hidden)
        self.c_proj = Linear(config.n_hidden, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.c_fc_1.weight.wd_mul = 2.0
        self.c_fc_2.weight.wd_mul = 2.0
        self.c_proj.weight.wd_mul = 2.0

    def forward(self, x):
        x = self.c_fc_1(x)
        x = F.relu(x).square()
        x = self.c_fc_2(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_1 = CausalSelfAttention(config)
        self.attn_2 = CausalSelfAttention(config)
        self.fnn = FeedForward(config)

    def forward(self, x):
        x = x + self.fnn(norm(x)) + self.attn_1(norm(x)) + self.attn_2(norm(x))
        return x

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 4282
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 32
    n_hidden: int = 32 * 4 # feedforward hidden size. Typically is set to `4 * n_embd`
    beta1: float = 0.9
    beta2: float = 0.95
    dropout: float = 0.0

class Palm(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = Linear(config.n_embd, config.vocab_size)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = norm(x)

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        else:
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        if not self.use_rope:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {Fore.WHITE}{Style.BRIGHT}{len(decay_params)}"
            f"{Style.RESET_ALL},",
            f"with {Fore.WHITE}{Style.BRIGHT}{num_decay_params:,}",
            "parameters"
        )

        print(
            f"num non-decayed parameter tensors: {Fore.WHITE}{Style.BRIGHT}{len(nodecay_params)}"
            f"{Style.RESET_ALL},",
            f"with {Fore.WHITE}{Style.BRIGHT}{num_nodecay_params:,}",
            "parameters"
        )

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        color = f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}" if use_fused == True else f"{Fore.LIGHTRED_EX}{Style.BRIGHT}"
        print(f"using fused AdamW: {color}{use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(self.config.beta1, self.config.beta2), fused=use_fused)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

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
