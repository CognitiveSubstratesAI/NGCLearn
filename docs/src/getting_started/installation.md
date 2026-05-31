# Installation

NGCLearn requires **Julia 1.12+** (it inherits NGCSimLib's `OncePerProcess`
floor).

## For development

NGCLearn depends on [NGCSimLib](https://github.com/CognitiveSubstratesAI/NGCSimLib)
via a `[sources]` path dev-link (`../NGCSimLib` in `Project.toml`), so clone both
side by side under the same parent directory:

```bash
git clone https://github.com/CognitiveSubstratesAI/NGCSimLib
git clone https://github.com/CognitiveSubstratesAI/NGCLearn
cd NGCLearn
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Dependencies

Through NGCSimLib, NGCLearn pulls in [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)
+ [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for the JIT / autodiff path.
These are heavy (LLVM/XLA) but precompile once.
