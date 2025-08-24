# Palm: Palm is a tree, not a language model.
An experimental language model architecture based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) project.

It's an experiment to try different improvements of transformers architecture. Some improvement has been brought about by the following techniques:
- Reuse a shared stack of layers across recursion steps inspired from Google's Mixture of Recursions [[paper](https://arxiv.org/pdf/2507.10524)]
- Use a novel attention mechanism which I call `Attention On Detail`
- Modernized architecture: Rotary embeddings, QK-Norm, and ReLU²
- Parallel layers proposed by Google's PaLM [[paper](https://arxiv.org/pdf/2204.02311)]
- SwiGLU in feed forward network. [[paper](https://arxiv.org/pdf/2002.05202)]
- Untie head from embedding
- The Muon optimizer [[writeup](https://kellerjordan.github.io/posts/muon)] [[repo](https://github.com/KellerJordan/Muon)]
- Linear layer factorization

As well as many minor optimizations.

## How does `Attention On Detail` works?
### It works by combining some very simple ideas.
- [Multi-Headed Causal Self-Attention (MHA)](https://arxiv.org/pdf/1706.03762)
- [Attention Free Transformer (AFT)](https://arxiv.org/pdf/2105.14103)
- [Linear Attention Mechanism (LAM)](https://arxiv.org/pdf/2007.14902)
- [Key-Value Transformer](https://arxiv.org/pdf/2305.19129)
- [Neural Attention](https://arxiv.org/pdf/2310.11398)
- [SwiGLU](https://arxiv.org/pdf/2002.05202)

### The idea is simple.
1. Initialize `qkv` FFN layers and pass `x` through them.
2. Use the `qkv` values in causal `AFT`/`LAM` to calculate attention scores.
3. Pass the result of that through a single-layered SwiGLU function.

### It has 3 core benefits.
1. Parameter efficient (only if FFNs are factorized else same number of parameters as standard `MHA`).
2. As compute efficient as `AFT` and `LAM`.
3. Very expressive.

## Citation

```
@software{Palm,
    author={Srijan Srivastava},
    title={Palm},
    url={https://github.com/SrijanSriv211/Palm},
    version={0.1.0},
    year = {2025}
}
```

<img src="img/rdr2.png" alt="lookwhosback" style="width:100%;">
