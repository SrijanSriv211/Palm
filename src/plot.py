import matplotlib.pyplot as plt
import warnings, torch, sys, os

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

def plot(title, plot_data, save_path):
	with plt.style.context("seaborn-v0_8-dark"):
		for param in ["figure.facecolor", "axes.facecolor", "savefig.facecolor"]:
			plt.rcParams[param] = "#030407"

		for param in ["text.color", "axes.labelcolor", "xtick.color", "ytick.color"]:
			plt.rcParams[param] = "0.9"

		plt.figure(figsize=(18, 8))

		for losses, label in plot_data:
			plt.plot(losses, label=label)

		plt.xlabel("iteration", fontsize=12)
		plt.ylabel("value", fontsize=12)
		plt.legend(fontsize=12)
		plt.title(title, fontsize=14)
		plt.savefig(save_path, bbox_inches="tight")
		plt.close()

if not os.path.isdir("res"):
	os.mkdir("res")

checkpoint = torch.load(sys.argv[1])
stats = checkpoint["stats"]

plot("eval loss", [(stats["train"], "train loss"), (stats["val"], "val loss")], f"res/eval.png")
plot("log loss", [(stats["eval"], "log loss")], f"res/log.png")
plot("lr", [(stats["lr"], "lr")], f"res/lr.png")
