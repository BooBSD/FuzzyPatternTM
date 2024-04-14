# Fuzzy Patterns Tsetlin Machine

Experimental version of Tsetlin Machine.
The changes compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl) are located in the following functions: `check_clause()` and `feedback!()`.
Please, see the comments.

Key features compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl)
---------------------------------------------------------------------------

  - New hyperparameter `LF` that sets the number of literal misses allowed for the clause.
  - Good accuracy and learning speed for small models. Achieves up to 98.49% accuracy on MNIST and 89.27% accuracy on Fashion MNIST using a model with 20 clauses per class (10 positive + 10 negative). The original Tsetlin Machine from 2018 achieves approximately the same accuracy but with 2000 clauses per class.
  - The accuracy and learning speed of large models (2048 clauses per class) are not very good. We need to delve deeper into this issue.
  - Blazingly fast batch inference is broken. :(
  - Binomial combinatorial merging degrades model accuracy. :(


How to run MNIST and Fashion MNIST examples
-------------------------------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Run `julia --project=. -O3 -t 32,1 --gcthreads=32,1 mnist_20.jl` where `32` is the number of your logical CPU cores.
