from colorama import Style, Fore, init
from encoder import Encoder
from model import sample
import warnings, argparse, torch, sys, os

# supress pytorch's future warning:
# You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly.
# It is possible to construct malicious pickle data which will execute arbitrary code during unpickling
# (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details).
# In a future release, the default value for `weights_only` will be flipped to `True`.
# This limits the functions that could be executed during unpickling.
# Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`.
# We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file.
# Please open an issue on GitHub for any issues related to this experimental feature.
warnings.filterwarnings("ignore", category=FutureWarning)
init(autoreset = True)

# check for the input source and append all the input texts to the inputs list.
def load_text(filepath):
    if os.path.isfile(filepath) == False:
        print(f"{filepath}: no such file or directory")
        sys.exit()

    with open(filepath, "r", encoding="utf-8") as f:
        return [i.strip() for i in f.readlines()]
    return []

parser = argparse.ArgumentParser(description="A powerful text encryption and decryption program.")
parse_prompt = parser.add_mutually_exclusive_group()
parser.add_argument("--model", "-i", help="model path", required=True)
parser.add_argument("--encoder", "-e", help="encoder path", required=True)
parser.add_argument("--length", "-l", help="output length", type=int, default=256)
parser.add_argument("--temperature", "-t", help="output temperature", type=float, default=0.8)
parser.add_argument("--top_k", "-f", help="output top_k", type=int, default=None)
parser.add_argument("--stream", "-s", help="stream output", type=bool, default=False)
parse_prompt.add_argument("--text_prompt", "-T", help="Text input from the command line.")
parse_prompt.add_argument("--file_prompt", "-F", help="Takes a text file as an input.")
args = parser.parse_args()

enc = Encoder()
enc.load(args.encoder)

s = sample(enc=enc)
s.load(torch.load(args.model), True)

# load text from the text file.
text = [args.text_prompt] if args.text_prompt else load_text(args.file_prompt) if args.file_prompt else [None]

for txt in text:
	enctxt = enc.encode(txt, allowed_special="all") if txt != None else txt
	out = enc.decode(s.generate(enctxt, length=args.length, temperature=args.temperature, top_k=args.top_k, stream=args.stream))
	print(f"{Fore.WHITE}{Style.DIM}```\n{out}\n```\n") if not args.stream else None
