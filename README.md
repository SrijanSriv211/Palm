# Palm: Palm is a tree, not a language model.
An experimental language model architecture based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) project.

It's an experiment to try different improvements of transformers architecture. Some improvement has been brought about by the following techniques:
- Modernized architecture: Rotary embeddings, QK-Norm, and ReLU²
- Untie head from embedding
- SwiGLU in feed forward network.
- Parallel layers proposed by Google's PaLM
- Using a novel attention mechanism which I call `Attention On Detail`.

As well as many minor optimizations.

## How does `Attention On Detail` works?
It works by combining 3 ideas.
- [Multi-Headed Causal Self-Attention (MHA)](https://arxiv.org/pdf/1706.03762)
- [Attention Free Transformer (AFT)](https://arxiv.org/pdf/2105.14103)
- A simple fourier series based equation `a*sin(x) + b*cos(x) + c*sin(x)*cos(x)` where `x` is normalized between `[-pi, pi]`

The idea is simple.
- Replace `Linear layers` with an `AFT` for each `q`, `k` & `v` in the `MHA`.
- In `AFT`, generate 3 values, `a`, `b` and `c` from 3 different fourier series equations.
- Compute output the `a`, `b` & `c` values in each `AFT`.
- Now use those `q`, `k` & `v` values to calculate the attention score in the `MHA`

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
