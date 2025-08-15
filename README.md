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
### It works by combining 5 simple ideas.
1. [Multi-Headed Causal Self-Attention (MHA)](https://arxiv.org/pdf/1706.03762)
2. [Attention Free Transformer (AFT)](https://arxiv.org/pdf/2105.14103)
3. [Key-Value Transformer](https://arxiv.org/pdf/2305.19129)
4. [Neural Attention](https://arxiv.org/pdf/2310.11398)
5. [SwiGlu](https://arxiv.org/pdf/2002.05202)

### The idea is simple.
1. Initialize only `kv` FFN layers and pass `x` through them.
2. Let `q = k` and use `AFT` equation to calculate a new query value (`q`).
3. Use the `qkv` values in an oversimplified, element-wise variant of scaled dot product causal self-attention.
4. Pass the result of that through a single-layered SwiGLU function.

### It has 3 benefits.
1. Parameter efficient (only if FFNs are factorized else same number of parameters as standard `MHA`).
2. As compute efficient as `AFT`.
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
