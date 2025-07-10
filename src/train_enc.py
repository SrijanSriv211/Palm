from encoder import Encoder
import argparse
enc = Encoder()

# python src/train_enc.py -i data/base/data.txt -o bin/cl4k.bin -v 4096
parser = argparse.ArgumentParser(description="A powerful text encryption and decryption program.")
parser.add_argument("-i", help="dataset path", required=True)
parser.add_argument("-o", help="output path", required=True)
parser.add_argument("-v", help="vocab size", type=int, required=True)
parser.add_argument("-r", help="text range", type=int, default=100_000_000)
parser.add_argument("-s", help="special tokens", type=list, default=["<|sot|>", "<|eot|>", "<|pad|>", "<|sep|>"])
args = parser.parse_args()

CONFIG = {
	"dataset_path": args.i,
	"outpath": args.o,
	"vocab_size": args.v - len(args.s), # remove len of special tokens to get vocab size for merging
	"text_range": args.r,
	"special_tokens": args.s
}

#* set `vocab_size` in `config.json` 4096
enc.train(CONFIG["dataset_path"], CONFIG["vocab_size"], text_range=CONFIG["text_range"])
enc.register_special_tokens(*CONFIG["special_tokens"])
enc.save(CONFIG["outpath"])
