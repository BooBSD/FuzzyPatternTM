# Fuzzy-Pattern Tsetlin Machine

Experimental version of Tsetlin Machine.
The changes compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl) are located in the following functions: `check_clause()`, `feedback!()` and `train!()`.
Please, see the comments.

**FashionMNIST** peak test accuracy using *convolutional* preprocessing:

- 2 clauses per class: **92.20%**
- 20 clauses per class: **93.41%**
- 2000 clauses per class: **94.10%**


Here is the tiny **20-clause** model training result for the **MNIST** dataset:

<img width="698" alt="Experimental Fuzzy Patterns Tsetlin Machine MNIST accuracy 98.56%" src="https://github.com/user-attachments/assets/05768a26-036a-40ce-b548-95925e96a01d">

Key features compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl)
---------------------------------------------------------------------------

  - New hyperparameter `LF` that sets the number of literal misses allowed for the clause. The special case `LF = 1` corresponds to the same internal logic used in the [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl) library.
  - Good accuracy and learning speed for small models. Achieves up to **98.56%** peak accuracy on **MNIST** and **89.67%** peak accuracy on **Fashion MNIST** using a model with **20 clauses** per class (10 positive + 10 negative). The original Tsetlin Machine from 2018 achieves approximately the same accuracy but with **2000** clauses per class.

How to run MNIST example
------------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Run `julia --project=. -O3 -t 32 mnist.jl` where `32` is the number of your logical CPU cores.

How to run FashionMNIST example using convolutional preprocessing
-----------------------------------------------------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Run `julia --project=. -O3 -t 32 fmnist_conv.jl` where `32` is the number of your logical CPU cores.
