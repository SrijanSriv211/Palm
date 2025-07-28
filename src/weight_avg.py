from colorama import Style, Fore, init
from collections import OrderedDict
import warnings, argparse, torch, os

# supress pytorch's future warning:
# You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly.
# It is possible to construct malicious pickle data which will execute arbitrary code during unpickling
# (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details).
# In a future release, the default value for `weights_only` will be flipped to `True`.
# This limits the functions that could be executed during unpickling.
# Arbitrary objects will no longer be allowed to be loaded via this mode
# unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`.
# We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file.
# Please open an issue on GitHub for any issues related to this experimental feature.
warnings.filterwarnings("ignore", category=FutureWarning)
init(autoreset=True)

parser = argparse.ArgumentParser(description="Average checkpoint weights")
parser.add_argument("-n", help="load last `n` checkpoints", type=int, required=True)
parser.add_argument("-s", help="step diff b/w each checkpoint", type=int, required=True)
parser.add_argument("-p", help="checkpoints path", type=str, required=True)
parser.add_argument("-o", help="outpath for the averaged model", type=str, required=True)
args = parser.parse_args()

# check for arguments validity
if not os.path.isdir(args.p):
    print(f"error: checkpoint path '{args.p}' not found or is not a directory.")
    exit(1)

if args.n <= 0:
    print(f"error: number of checkpoints `n` ({args.n}) must be positive.")
    exit(1)

if args.s <= 0:
    print(f"error: step diff `s` ({args.s}) must be positive.")
    exit(1)

print("loading checkpoints")
all_checkpoints = [int(i[1:-4]) for i in os.listdir(args.p) if i.startswith("s") and i.endswith(".bin")]
all_checkpoints.sort()
all_checkpoints.reverse()

needed_checkpoints = [f"{args.p}/s{i}.bin" for i in all_checkpoints if i % args.s == 0][:args.n]
del all_checkpoints

# load all state dicts and hyperparams
state_dicts = []
hyperparams = None
optimizer = None
device = None
stats = None

for i in needed_checkpoints:
	print(f"{Fore.WHITE}{Style.BRIGHT}>", i)
	checkpoint = torch.load(i)
	if hyperparams is None:
		hyperparams = checkpoint["hyperparams"]
		optimizer = checkpoint["optimizer"]
		device = checkpoint["device"]
		stats = checkpoint["stats"]

	# remove `_orig_mod.` prefix from state_dict (if it's there)
	state_dict = checkpoint["model"]
	unwanted_prefix = '_orig_mod.'

	for k, v in list(state_dict.items()):
		if k.startswith(unwanted_prefix):
			state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
	state_dicts.append(state_dict)
del needed_checkpoints

# calculate avg
print("averaging checkpoints")
avg_state_dict = None
for state_dict in state_dicts:
	if avg_state_dict is None:
		avg_state_dict = OrderedDict()
		for k, v in state_dict.items():
			avg_state_dict[k] = v.clone()

	# accumulate weights
	else:
		for k, v in state_dict.items():
			avg_state_dict[k].add_(v)

for k in avg_state_dict.keys():
    avg_state_dict[k] = avg_state_dict[k].div(args.n)

# save model
print("saving averaged model")
torch.save({
	"model": avg_state_dict,
	"optimizer": optimizer,
	"hyperparams": hyperparams,
	"device": device,
	"stats": stats
}, args.o)
