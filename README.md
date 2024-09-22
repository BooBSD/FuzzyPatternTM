# Fuzzy Patterns Tsetlin Machine

Experimental version of Tsetlin Machine.
The changes compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl) are located in the following functions: `check_clause()`, `feedback!()` and `train!()`.
Please, see the comments.

Here is the tiny 20-clause model training result for the `MNIST` dataset:

<img width="698" alt="Experimental Tsetlin Machine MNIST accuracy 98.52%" src="https://github.com/user-attachments/assets/f4b52bd1-4700-450c-95f5-984247a73e02">

Key features compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl)
---------------------------------------------------------------------------

  - New hyperparameter `LF` that sets the number of literal misses allowed for the clause.
  - Good accuracy and learning speed for small models. Achieves up to **98.52%** accuracy on **MNIST** and **89.67%** accuracy on **Fashion MNIST** using a model with **20 clauses** per class (10 positive + 10 negative). The original Tsetlin Machine from 2018 achieves approximately the same accuracy but with **2000** clauses per class.
  - The accuracy and learning speed of large models (2048 clauses per class) are not very good. We need to delve deeper into this issue.
  - Blazingly fast batch inference is broken. :(
  - Binomial combinatorial merging degrades model accuracy. :(

Strange training behavior observed in large models
--------------------------------------------------

Large models (512+ clauses per class) experience accuracy plateaus after a certain number of epochs:

<img width="706" alt="Accuracy stopped" src="https://github.com/BooBSD/FuzzyPatternsTM/assets/48304/acd0304a-9f7f-487a-b502-142cb4d3e05f">

How to run MNIST and Fashion MNIST examples
-------------------------------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Run `julia --project=. -O3 -t 32 --gcthreads=32,1 mnist.jl` where `32` is the number of your logical CPU cores.
