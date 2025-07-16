from model import Config, Palm, sample
from colorama import Style, Fore, init
from encoder import Encoder
from pathlib import Path
from contextlib import nullcontext
from rich.progress import track
import torch, random, numpy, time, math, os
import torch.amp, json, regex, sys
torch._inductor.config.coordinate_descent_tuning = True
torch._dynamo.config.compiled_autograd = True

# load config
init(autoreset=True)
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "script/config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
	CONFIG = json.load(f)

# set device
device = ("cuda" if torch.cuda.is_available() else "cpu") if CONFIG["device"] == "auto" else CONFIG["device"]
# init seed
if CONFIG["seed"] != "auto":
	torch.manual_seed(CONFIG["seed"])
	numpy.random.seed(CONFIG["seed"])
	random.seed(CONFIG["seed"])

# save the text in a text file
ansi_escape = regex.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
def print0(*text, println=True):
    print(*text) if println else None

    # save cleaned text to the file
    with open(os.path.join(Path(CONFIG["save_path"]).parent.name, "out.txt"), "a", encoding="utf-8") as f:
        f.write(" ".join(tuple(ansi_escape.sub('', part) for part in text)) + "\n")

def calc_total_time(seconds):
    # separate the integer part (for hours, minutes, and seconds) from the fractional part (for milliseconds)
    sec_int, millis = divmod(seconds, 1)
    millis = int(millis * 1000) # convert the fractional part to milliseconds

    min, sec = divmod(int(sec_int), 60)
    hour, min = divmod(min, 60)
    hours, minutes, seconds = int(hour), int(min), int(sec)

    t = [
        f"{hours} hour" + ("s" if hours > 1 else "") if hours > 0 else None,
        f"{minutes} minute" + ("s" if minutes > 1 else "") if minutes > 0 else None,
        f"{seconds} second" + ("s" if seconds > 1 else "") if seconds > 0 else None,
        f"{millis} ms" if millis > 0 else None
    ]
    t = list(filter(None, t))

    return ", ".join(t) if t else "0 seconds"

def init_model(checkpoint=None):
	# print the device
	print0(f"```config.json\n{json.dumps(CONFIG, indent=4)}\n```", println=False)
	print0("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{device}", f"{Fore.WHITE}{Style.BRIGHT}({torch.initial_seed()})")

	# load stats
	stats = checkpoint["stats"] if checkpoint is not None and "stats" in checkpoint.keys() else {
		"steps": 0,
		"train": [],
		"eval": [],
		"val": [],
		"lr": []
	}

	# load hyperparams
	hyperparams = dict(dropout=CONFIG["dropout"])
	# read off the created CONFIG params, so we can store them into checkpoint correctly
	for k in ["n_layer", "n_head", "n_embd", "n_hidden", "block_size", "vocab_size", "d_factor"]:
		hyperparams[k] = CONFIG[k]
	# automatically set `n_hidden` for feedforward network if not set already
	if any([hyperparams["n_hidden"] == i for i in ["4x_embd", "auto", None]]):
		hyperparams["n_hidden"] = hyperparams["n_embd"] * 4

	# create an instance of Palm
	conf = Config(**hyperparams)
	model = Palm(conf)
	# remove `_orig_mod.` prefix from state_dict (if it's there)
	if checkpoint is not None:
		state_dict = checkpoint["model"]
		unwanted_prefix = '_orig_mod.'
		for k, v in list(state_dict.items()):
			if k.startswith(unwanted_prefix):
				state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
		# load the state dict
		model.load_state_dict(state_dict)
	model.to(device)

	# optimizers!
	# start with all of the candidate parameters, filter out those that do not require grad
	param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
	# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
	# i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
	decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
	nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
	optim_groups = [
		{'params': decay_params, 'weight_decay': CONFIG["weight_decay"]},
		{'params': nodecay_params, 'weight_decay': 0.0}
	]
	optimizer = torch.optim.AdamW(optim_groups, lr=CONFIG["learning_rate"], betas=(CONFIG["beta1"], CONFIG["beta2"]), fused=True)
	return model, optimizer, hyperparams, stats

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
	if not CONFIG["decay_lr"]:
		return CONFIG["learning_rate"]

	# 1) linear warmup for warmup_iters steps
	elif it < CONFIG["warmup_iters"]:
		return CONFIG["learning_rate"] * (it + 1) / (CONFIG["warmup_iters"] + 1)

	# 2) if it > lr_decay_iters, return min learning rate
	elif it > CONFIG["lr_decay_iters"]:
		return CONFIG["min_lr"]

	# 3) in between, use cosine decay down to min learning rate
	decay_ratio = (it - CONFIG["warmup_iters"]) / (CONFIG["lr_decay_iters"] - CONFIG["warmup_iters"])

	assert 0 <= decay_ratio <= 1
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
	return CONFIG["min_lr"] + coeff * (CONFIG["learning_rate"] - CONFIG["min_lr"])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(eval_iters, model, get_batch):
	out = {}
	model.eval()
	for split in ["train", "val"]:
		losses = torch.zeros(eval_iters)
		for k in track(range(eval_iters), description=f"{Fore.WHITE}{Style.BRIGHT}calc {Fore.WHITE}{Style.DIM}{split} loss{Style.RESET_ALL}"):
			X, Y = get_batch(split)
			with ctx:
				_, loss = model(X, Y)

			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

# estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
def estimate_mfu(fwdbwd_per_iter, model, dt):
	# first estimate the number of flops we do per iteration.
	# see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
	N = sum(p.numel() for p in model.parameters())
	L, H, Q, T = CONFIG["n_layer"], CONFIG["n_head"], CONFIG["n_embd"]//CONFIG["n_head"], CONFIG["block_size"]
	flops_per_token = 6*N + 12*L*H*Q*T
	flops_per_fwdbwd = flops_per_token * T
	flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
	# express our flops throughput as ratio of A100 bfloat16 peak flops
	flops_achieved = flops_per_iter * (1.0/dt) # per second
	flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
	mfu = flops_achieved / flops_promised
	return mfu

class dataloader:
	def __init__(self, path, isfile=True, t_in_mem=72_000_000, reload=True):
		self.path = path
		self.files = [path] if isfile else [os.path.join(path, i) for i in os.listdir(path)]
		self.t_in_mem = t_in_mem # tokens in memory
		self.reload_interval = None
		self.reload = reload

	# get total number of tokens
	def get_tok_count(self):
		n_toks = sum([numpy.memmap(file, dtype=numpy.int16, mode="r").size for file in self.files])

		# calculate when to reload the dataset
		reload_interval = round(self.t_in_mem/n_toks * CONFIG["max_iters"])
		if self.reload and self.t_in_mem is not None and reload_interval < CONFIG["max_iters"]:
			self.reload_interval = reload_interval
			print("dataset from", f"{Fore.WHITE}{Style.DIM}`{self.path}`", "will reload every", f"{Fore.WHITE}{Style.BRIGHT}{self.reload_interval}", "steps")

		return n_toks

	def load_dataset(self):
		self.data = []
		for f in random.sample(self.files, k=len(self.files)):
			# reshape data with 1024 since that is the length used in `prepare_data.py` while preparing datasets
			data = numpy.memmap(f, dtype=numpy.int16, mode="r")
			data = data.reshape((data.size // 1024, 1024))
			self.data.extend(data)
			del data

			if self.t_in_mem is not None and round(sum(i.size for i in self.data) / self.t_in_mem, 3) >= 0.999:
				self.data = numpy.array(self.data, dtype=numpy.int16)
				ix = numpy.random.randint(self.data.size - self.t_in_mem) // 1024
				self.data = self.data[ix : ix + (self.t_in_mem // 1024)]
				break

		if isinstance(self.data, list):
			self.data = numpy.array(self.data, dtype=numpy.int16)

	def next_batch(self, it=None):
		if self.reload and self.reload_interval is not None and it is not None and (it + 1) % self.reload_interval == 0:
			print("reloading dataset from", f"{Fore.WHITE}{Style.DIM}`{self.path}`")
			self.load_dataset()

		ix = torch.randint(self.data.shape[1] - CONFIG["block_size"], (CONFIG["batch_size"],))
		iy = torch.randint(self.data.shape[0], (CONFIG["batch_size"],))

		# create a mask where random positions of `ix` will be set to zero.
		# sample n unique positions in [0..batch_size):
		mask = torch.randperm(CONFIG["batch_size"])[:CONFIG["batch_size"]//2] # shape (CONFIG["batch_size"]//2,)
		ix[mask] = 0 # set those ix entries to 0

		# get x, y batches
		x = torch.stack([torch.from_numpy((self.data[j][i : i + CONFIG["block_size"]]).astype(numpy.int64)) for i, j in zip(ix, iy)])
		y = torch.stack([torch.from_numpy((self.data[j][i+1 : i+1 + CONFIG["block_size"]]).astype(numpy.int64)) for i, j in zip(ix, iy)])
		x, y = x.to(device), y.to(device)
		return x, y

# init model and optimizer
model, optimizer, hyperparams, stats = init_model(torch.load(CONFIG["init_from"][11:]) if CONFIG["init_from"].startswith("pretrained,") else None)
# "float32", "bfloat16", or "float16", the latter will auto implement a GradScaler
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=ptdtype)
torch.set_float32_matmul_precision("high")

# load train and val data
train_data_loader = dataloader(CONFIG["train_data"], CONFIG["load_from_file"])
val_data_loader = dataloader(CONFIG["val_data"], CONFIG["load_from_file"])
train_data_loader.load_dataset()
val_data_loader.load_dataset()
# simple lambda function for `estimate_loss` function
get_batch = lambda x: train_data_loader.next_batch() if x == "train" else val_data_loader.next_batch()

# print the number of tokens
num_train_toks = train_data_loader.get_tok_count()
num_val_toks = val_data_loader.get_tok_count()
print0(f"{Fore.WHITE}{Style.BRIGHT}{((num_train_toks + num_val_toks)/1e6)}M", "total tokens")
print0(
	f"{Fore.WHITE}{Style.BRIGHT}{(num_train_toks/1e6)}M", "train tokens,", f"{Fore.WHITE}{Style.BRIGHT}{(num_val_toks/1e6)}M", "val tokens",
	f"   {Fore.WHITE}{Style.DIM}(using train tokens as val tokens)" if CONFIG["train_data"] == CONFIG["val_data"] else ""
)
del num_train_toks, num_val_toks

# report number of parameters
print0(f"{Fore.WHITE}{Style.BRIGHT}{sum(p.numel() for p in model.parameters())/1e6}M", "parameters")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=False)

# compile the model
if CONFIG["compile"]:
	print0(f"compiling the model... {Fore.WHITE}{Style.DIM}(takes a ~minute)")
	model = torch.compile(model) # requires PyTorch 2.0

# training loop
start_time = time.time()
eval_t0 = time.time()
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0

# load encoder and sample constructor
# for sampling text using training
enc = Encoder()
enc.load(CONFIG["encoder_path"])
training_sample = sample()

def get_trained_model(model, optimizer):
	return {
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"hyperparams": hyperparams,
		"device": device,
		"stats": stats
	}

# write checkpoints
def save_checkpoint(model, optimizer):
	if CONFIG["checkpoints"] == None or stats["steps"] <= 0 or stats["steps"] % CONFIG["checkpoints"]["interval"] != 0: return
	if not os.path.isdir(CONFIG["checkpoints"]["path"]): os.mkdir(CONFIG["checkpoints"]["path"])
	print0(f"saved checkpoint at step {Fore.WHITE}{Style.BRIGHT}{stats["steps"]}")
	torch.save(get_trained_model(model, optimizer), f"{CONFIG["checkpoints"]["path"]}/s{stats["steps"]}.bin")

# generate some sample text
def sample_output(model, optimizer):
	if CONFIG["sample_interval"] == None or stats["steps"] <= 0 or stats["steps"] % CONFIG["sample_interval"] != 0: return
	training_sample.load(get_trained_model(model, optimizer), True)
	print0(f"{Fore.WHITE}{Style.DIM}```s{stats["steps"]}.bin\n{enc.decode(training_sample.generate(None, length=CONFIG["block_size"]))}\n```")

# evaluate the loss on train/val sets
def log_eval_loss():
	if stats["steps"] <= 0 or stats["steps"] % CONFIG["eval_interval"] != 0: return
	global eval_t0
	# timing and logging
	losses = estimate_loss(CONFIG["eval_iters"], model, get_batch)
	eval_t1 = time.time()
	eval_dt = eval_t1 - eval_t0
	eval_t0 = eval_t1
	print0(
		f"{Fore.WHITE}{Style.BRIGHT}step",
		f"{Fore.WHITE}{Style.DIM}[{stats["steps"]}/{CONFIG["max_iters"]}]"
		f"{Fore.RESET}{Style.RESET_ALL}:",
		f"train loss {Fore.WHITE}{Style.BRIGHT}{losses["train"]:.4f}"
		f"{Fore.RESET}{Style.RESET_ALL},",
		f"val loss {Fore.WHITE}{Style.BRIGHT}{losses["val"]:.4f}"
		f"{Fore.RESET}{Style.RESET_ALL},",
		f"lr {Fore.WHITE}{Style.BRIGHT}{lr:.7f}"
		f"{Fore.RESET}{Style.RESET_ALL},",
		f"time took {Fore.WHITE}{Style.DIM}{calc_total_time(eval_dt)}"
	)
	stats["train"].append(losses["train"])
	stats["val"].append(losses["val"])

def log_loss():
	global local_iter_num, running_mfu, t0
	# timing and logging
	t1 = time.time()
	dt = t1 - t0
	t0 = t1

	if stats["steps"] % CONFIG["log_interval"] != 0:
		return

	# get loss as float. note: this is a CPU-GPU sync point
	# scale up to undo the division above, approximating the true total loss (exact would have been a sum)
	lossf = loss.item() * CONFIG["gradient_accumulation_steps"]

	if local_iter_num >= 5: # let the training loop settle a bit
		# https://github.com/karpathy/nanoGPT/pull/527/files
		mfu = estimate_mfu(CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"] * CONFIG["log_interval"], model, dt)
		running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

	toks_per_sec = (CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"] * CONFIG["block_size"]) / dt
	print0(
		f"{Fore.WHITE}{Style.BRIGHT}iter",
		f"{Fore.WHITE}{Style.DIM}[{stats["steps"]}/{CONFIG["max_iters"]}]"
		f"{Fore.RESET}{Style.RESET_ALL}:",
		f"loss {Fore.WHITE}{Style.BRIGHT}{lossf:.4f}"
		f"{Fore.RESET}{Style.RESET_ALL},",
		f"mfu {Fore.WHITE}{Style.BRIGHT}{running_mfu*100:.2f}"
		f"{Fore.RESET}{Style.RESET_ALL},",
		f"dt {Fore.WHITE}{Style.DIM}{calc_total_time(dt)}"
		f"{Fore.RESET}{Style.RESET_ALL},",
		f"tok/s {Fore.WHITE}{Style.DIM}{toks_per_sec:.2f}"
	)
	stats["eval"].append(lossf)

# forward backward update, with optional gradient accumulation to simulate larger batch size
# and using the GradScaler if data type is float16
def train_model():
	global optimizer, model, get_batch
	for _ in range(CONFIG["gradient_accumulation_steps"]):
		# immediately async prefetch next batch while model is doing the forward pass on the GPU
		X, Y = get_batch("train")

		with ctx:
			_, loss = model(X, Y)
			loss = loss / CONFIG["gradient_accumulation_steps"] # scale the loss to account for gradient accumulation

		# backward pass, with gradient scaling if training in fp16
		scaler.scale(loss).backward()

	# clip the gradient
	if CONFIG["grad_clip"] != 0.0:
		scaler.unscale_(optimizer)
		torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

	# step the optimizer and scaler if training in fp16
	scaler.step(optimizer)
	scaler.update()

	# flush the gradients as soon as we can, no need for this memory anymore
	optimizer.zero_grad(set_to_none=True)
	model.zero_grad(set_to_none=True)
	return loss

# warmup the training kernels
print0(f"warming up training kernels... {Fore.WHITE}{Style.DIM}(takes a ~minute)")
for _ in range(100):
	train_model()

# start training the model
print0("started training")
for _ in range(CONFIG["max_iters"]):
	# determine and set the learning rate for this iteration
	lr = get_lr(stats["steps"])
	for group in optimizer.param_groups:
		group["lr"] = lr
	stats["lr"].append(lr)

	# validation section
	# save checkpoint and log sample and eval loss
	save_checkpoint(model, optimizer)
	sample_output(model, optimizer)
	log_eval_loss()

	# training section
	loss = train_model()

	# logging
	log_loss()
	stats["steps"] += 1
	local_iter_num += 1

print0("total time:", calc_total_time(time.time() - start_time))
torch.save(get_trained_model(model, optimizer), CONFIG["save_path"])
