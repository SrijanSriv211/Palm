# Palm: Palm is a tree, not a language model.
An experimental language model architecture based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) project.

It's an experiment to try different improvements of transformers architecture. Some improvement has been brought about by the following techniques:
- Reuses a shared stack of layers across recursion steps inspired from Google's Mixture of Recursions [[paper](https://arxiv.org/pdf/2507.10524)]
- Using a novel attention mechanism which I call `Attention On Detail`.
- Modernized architecture: Rotary embeddings, QK-Norm, and ReLU²
- Parallel layers proposed by Google's PaLM [[paper](https://arxiv.org/pdf/2204.02311)]
- SwiGLU in feed forward network. [[paper](https://arxiv.org/pdf/2002.05202)]
- Untie head from embedding
- The Muon optimizer [[writeup](https://kellerjordan.github.io/posts/muon)] [[repo](https://github.com/KellerJordan/Muon)]
- Factorization

As well as many minor optimizations.

## How does `Attention On Detail` works?
It works by combining 3 ideas.
- [Multi-Headed Causal Self-Attention (MHA)](https://arxiv.org/pdf/1706.03762)
- [Attention Free Transformer (AFT)](https://arxiv.org/pdf/2105.14103)
- A simple fourier series based equation, note that `x` is the input and is normalized between `[-pi, pi]`
```python
a1*sin(x) + a2*cos(x) + a3*sin(x)*cos(x) + a4*sin(2*x) + a5*cos(2*x) + a6*sin(2*x)*cos(2*x) + a7*sin(2*x)*cos(x) + a8*sin(x)*cos(2*x) + a9*sin(3*x) + a10*cos(3*x) + a11*sin(3*x)*cos(3*x) + a12*sin(3*x)*cos(2*x) + a13*sin(3*x)*cos(x) + a14*sin(2*x)*cos(3*x) + a15*sin(x)*cos(3*x) + a16*sin(4*x) + a17*cos(4*x) + a18*sin(4*x)*cos(4*x) + a19*sin(4*x)*cos(3*x) + a20*sin(4*x)*cos(2*x) + a21*sin(4*x)*cos(x) + a22*sin(3*x)*cos(4*x) + a23*sin(2*x)*cos(4*x) + a24*sin(x)*cos(4*x)
```

The idea is simple.
- In `MHA`, replace `qkv` `Linear layers` with an `AFT`.
- In `AFT`, replace `abc` `Linear layers` with 3 fourier series equations noted above.
- Use `AFT-simple` equation stated in Apple's research paper to process `abc` values.
- At last, use the `qkv` values to calculate the attention score in the `MHA`.

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
