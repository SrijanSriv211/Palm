from utils import calc_total_time, kprint
from model import Config, Palm, sample
from colorama import Style, Fore, init
from encoder import Encoder
from contextlib import nullcontext
from rich.progress import track
from pathlib import Path
import torch, random, numpy, time, math, os
import torch._inductor.config as config
import torch.amp, json, sys

init(autoreset=True)

# load config
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "script/config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
	CONFIG = json.load(f)
model_log_path = os.path.join(Path(CONFIG["save_path"]).parent.name, "out.txt")

# set device
device = ("cuda" if torch.cuda.is_available() else "cpu") if CONFIG["device"] == "auto" else CONFIG["device"]
# init seed
if CONFIG["seed"] != "auto":
	torch.manual_seed(CONFIG["seed"])
	numpy.random.seed(CONFIG["seed"])
	random.seed(CONFIG["seed"])

def init_model(checkpoint=None):
	global model_log_path
	# if model is init from scratch then create an empty model log file.
	if checkpoint is None:
		with open(model_log_path, "w", encoding="utf-8") as f:
			f.write("")

	# print the device
	kprint(f"```config.json\n{json.dumps(CONFIG, indent=4)}\n```", filename=model_log_path, println=False)
	kprint("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{device}", f"{Fore.WHITE}{Style.BRIGHT}({torch.initial_seed()})", filename=model_log_path)

	# load stats
	stats = checkpoint["stats"] if checkpoint is not None and "stats" in checkpoint.keys() else {
		"iter_num": 0,
		"train": [],
		"eval": [],
		"val": [],
		"mfu": [],
		"lr": []
	}

	# load hyperparams
	hyperparams = dict(dropout=CONFIG["dropout"])
	# read off the created CONFIG params, so we can store them into checkpoint correctly
	for k in ["n_layer", "n_head", "n_embd", "n_hidden", "block_size", "vocab_size", "d_factor", "beta1", "beta2"]:
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

	# optimizer
	optimizer = model.configure_optimizers(CONFIG["weight_decay"], CONFIG["learning_rate"], CONFIG["device"])
	optimizer.load_state_dict(checkpoint["optimizer"]) if checkpoint is not None else None

	# crop down the model block size if desired, using model surgery
	if CONFIG["block_size"] < hyperparams["block_size"]:
		model.crop_block_size(CONFIG["block_size"])
		hyperparams["block_size"] = CONFIG["block_size"] # so that the checkpoint will have the right value

	return model, optimizer, hyperparams, stats

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < CONFIG["warmup_iters"]:
        return CONFIG["learning_rate"] * (it + 1) / (CONFIG["warmup_iters"] + 1)

    # 2) if it > lr_decay_iters, return min learning rate
    if it > CONFIG["lr_decay_iters"]:
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
				logits, loss = model(X, Y)

			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

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
kprint("using RoPE:", f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}True", filename=model_log_path)

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
kprint(f"{Fore.WHITE}{Style.BRIGHT}{((num_train_toks + num_val_toks)/1e6)}M", "total tokens", filename=model_log_path)
kprint(
	f"{Fore.WHITE}{Style.BRIGHT}{(num_train_toks/1e6)}M", "train tokens,", f"{Fore.WHITE}{Style.BRIGHT}{(num_val_toks/1e6)}M", "val tokens",
	f"   {Fore.WHITE}{Style.DIM}(using train tokens as val tokens)" if CONFIG["train_data"] == CONFIG["val_data"] else "", filename=model_log_path
)
del num_train_toks, num_val_toks

# report number of parameters
kprint(f"{Fore.WHITE}{Style.BRIGHT}{model.get_num_params()/1e6}M", "parameters", filename=model_log_path)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=False)

if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee

# compile the model
if CONFIG["compile"]:
	kprint(f"compiling the model... {Fore.WHITE}{Style.DIM}(takes a ~minute)", filename=model_log_path)
	model = torch.compile(model) # requires PyTorch 2.0

# training loop
X, Y = train_data_loader.next_batch() # fetch the very first batch
start_time = time.time()
eval_t0 = time.time()
t0 = time.time()
t2 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0 if stats["mfu"] == [] else stats["mfu"][-1]
training_loop = True

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

while training_loop:
	try:
		# determine and set the learning rate for this iteration
		lr = get_lr(stats["iter_num"]) if CONFIG["decay_lr"] else CONFIG["learning_rate"]
		for param_group in optimizer.param_groups:
			param_group["lr"] = lr
		stats["lr"].append(lr)

		# save checkpoint
		if CONFIG["checkpoints"] != None and stats["iter_num"] > 0 and stats["iter_num"] % CONFIG["checkpoints"]["interval"] == 0:
			if not os.path.isdir(CONFIG["checkpoints"]["path"]):
				os.mkdir(CONFIG["checkpoints"]["path"])

			kprint(f"saved checkpoint at step {Fore.WHITE}{Style.BRIGHT}{stats["iter_num"]}", filename=model_log_path)
			torch.save(get_trained_model(model, optimizer), f"{CONFIG["checkpoints"]["path"]}/s{stats["iter_num"]}.bin")

		# generate some sample text
		if CONFIG["sample_interval"] != None and stats["iter_num"] > 0 and stats["iter_num"] % CONFIG["sample_interval"] == 0:
			training_sample.load(get_trained_model(model, optimizer), True)

			for _ in range(CONFIG["sample_iters"]):
				out = enc.decode(training_sample.generate(None, length=CONFIG["block_size"]))
				kprint(f"{Fore.WHITE}{Style.DIM}```s{stats["iter_num"]}.bin\n{out}\n```", filename=model_log_path)

		# evaluate the loss on train/val sets and write checkpoints
		if stats["iter_num"] > 0 and stats["iter_num"] % CONFIG["eval_interval"] == 0:
			losses = estimate_loss(CONFIG["eval_iters"], model, get_batch)
			# timing and logging
			eval_t1 = time.time()
			eval_dt = eval_t1 - eval_t0
			eval_t0 = eval_t1

			kprint(
				f"{Fore.WHITE}{Style.BRIGHT}step",
				f"{Fore.WHITE}{Style.DIM}[{stats["iter_num"]}/{CONFIG["max_iters"]}]"
				f"{Fore.RESET}{Style.RESET_ALL}:",
				f"train loss {Fore.WHITE}{Style.BRIGHT}{losses["train"]:.4f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"val loss {Fore.WHITE}{Style.BRIGHT}{losses["val"]:.4f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"lr {Fore.WHITE}{Style.BRIGHT}{lr:.7f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"time took {Fore.WHITE}{Style.DIM}{calc_total_time(eval_dt)}",
				filename=model_log_path
			)

			stats["train"].append(losses["train"])
			stats["val"].append(losses["val"])

		# forward backward update, with optional gradient accumulation to simulate larger batch size
		# and using the GradScaler if data type is float16
		for micro_step in range(CONFIG["gradient_accumulation_steps"]):
			with ctx:
				logits, loss = model(X, Y)
				loss = loss / CONFIG["gradient_accumulation_steps"] # scale the loss to account for gradient accumulation

			# immediately async prefetch next batch while model is doing the forward pass on the GPU
			X, Y = train_data_loader.next_batch()
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

		# timing and logging
		t1 = time.time()
		dt = t1 - t0
		t0 = t1
		if stats["iter_num"] % CONFIG["log_interval"] == 0:
			t1 = time.time()
			dt2 = t1 - t2
			t2 = t1

			# get loss as float. note: this is a CPU-GPU sync point
			# scale up to undo the division above, approximating the true total loss (exact would have been a sum)
			lossf = loss.item() * CONFIG["gradient_accumulation_steps"]

			if local_iter_num >= 5: # let the training loop settle a bit
				mfu = model.estimate_mfu(CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"] * CONFIG["log_interval"], dt) # https://github.com/karpathy/nanoGPT/pull/527/files
				running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

			toks_per_sec = (CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"] * CONFIG["block_size"]) / dt
			kprint(
				f"{Fore.WHITE}{Style.BRIGHT}iter",
				f"{Fore.WHITE}{Style.DIM}[{stats["iter_num"]}/{CONFIG["max_iters"]}]"
				f"{Fore.RESET}{Style.RESET_ALL}:",
				f"loss {Fore.WHITE}{Style.BRIGHT}{lossf:.4f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"mfu {Fore.WHITE}{Style.BRIGHT}{running_mfu*100:.2f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"dt {Fore.WHITE}{Style.DIM}{calc_total_time(dt2)}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"tok/s {Fore.WHITE}{Style.DIM}{toks_per_sec:.2f}",
				filename=model_log_path
			)
			stats["mfu"].append(running_mfu)
			stats["eval"].append(lossf)

		stats["iter_num"] += 1
		local_iter_num += 1

		# termination conditions
		if stats["iter_num"] > CONFIG["max_iters"]:
			break

	except KeyboardInterrupt:
		print("type")
		print(f"{Fore.WHITE}{Style.BRIGHT}1. {Fore.WHITE}{Style.DIM}`y` {Style.RESET_ALL}to stop training.")
		print(f"{Fore.WHITE}{Style.BRIGHT}2. {Fore.WHITE}{Style.DIM}`n` {Style.RESET_ALL}to continue training.")
		print(f"{Fore.WHITE}{Style.BRIGHT}3. {Fore.WHITE}{Style.DIM}`s` {Style.RESET_ALL}to save model.")
		print(f"{Fore.WHITE}{Style.BRIGHT}4. {Fore.WHITE}{Style.DIM}`r` {Style.RESET_ALL}to reload config.json.")
		print(f"{Fore.WHITE}{Style.BRIGHT}5. {Fore.WHITE}{Style.DIM}`e` {Style.RESET_ALL}to force exit without save.")

		while True:
			inp = input("> ")

			if inp == "y":
				print(f"{Fore.RED}{Style.BRIGHT}early stopping.")
				training_loop = False
				break

			elif inp == "n":
				print(f"{Fore.GREEN}{Style.BRIGHT}continue training.")
				break

			elif inp == "s":
				print(f"{Fore.YELLOW}{Style.BRIGHT}saving model.")
				print("total time:", calc_total_time(time.time() - start_time))
				torch.save(get_trained_model(model, optimizer), CONFIG["save_path"])

			elif inp == "r":
				print(f"{Fore.YELLOW}{Style.BRIGHT}config.json{Style.RESET_ALL} reloaded.")
				with open(CONFIG_PATH, "r", encoding="utf-8") as f:
					CONFIG = json.load(f)

			if inp == "e":
				print(f"{Fore.RED}{Style.BRIGHT}forcing exit without save.")
				sys.exit()

			else:
				print(f"{Fore.RED}{Style.DIM}Wrong option.")

kprint("total time:", calc_total_time(time.time() - start_time), filename=model_log_path)
torch.save(get_trained_model(model, optimizer), CONFIG["save_path"])
