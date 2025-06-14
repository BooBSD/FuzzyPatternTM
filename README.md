# Fuzzy-Pattern Tsetlin Machine

## Abstract

The "*all-or-nothing*" clause evaluation strategy is a core mechanism in the Tsetlin Machine (TM) family of algorithms. In this approach, each clause—a logical pattern composed of binary literals mapped to input data—is disqualified from voting if even a single literal fails. Due to this strict requirement, standard TMs must employ thousands of clauses to achieve competitive accuracy.

This paper introduces the **Fuzzy-Pattern Tsetlin Machine** (FPTM), a novel variant where clause evaluation is fuzzy rather than strict. If some literals in a clause fail, the remaining ones can still contribute to the overall vote with a proportionally reduced score. As a result, each clause effectively consists of sub-patterns that adapt individually to the input, enabling more flexible, efficient, and robust pattern matching.

The proposed fuzzy mechanism significantly reduces the number of required clauses, memory footprint, and training time—while maintaining competitive accuracy. Experiments on the IMDB dataset demonstrate a median peak test accuracy of **90.15%** using only **one clause per class**, representing a **50× reduction** in both clause count and memory usage compared to CoalescedTM, a leading TM variant. The Tsetlin Automata state matrix fits within **50 KB**, enabling online training on modern microcontrollers. Training is **36× faster** than CoalescedTM on a single-threaded CPU, and **316× faster** with 32-thread parallelism. For instance, while CoalescedTM requires nearly **4 hours** to train for 1000 epochs, FPTM completes the same task in just **45 seconds** with comparable accuracy. Inference throughput reaches up to **34.5 million predictions per second** in batch mode, corresponding to **51.4 GB/s** on a desktop CPU.

Additional experiments on the FashionMNIST dataset with convolutional preprocessing yield similar results: **92.20%** peak test accuracy with only **2 clauses per class**, **93.41%** with 20 clauses per class, and **94.10%** with 2000 clauses per class.

Here is the tiny **20-clause** model training result for the **MNIST** dataset:
<img width="698" alt="Experimental Fuzzy Patterns Tsetlin Machine MNIST accuracy 98.56%" src="https://github.com/user-attachments/assets/05768a26-036a-40ce-b548-95925e96a01d">

Key features compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl)
---------------------------------------------------------------------------

  - New hyperparameter `LF` that sets the number of literal misses allowed for the clause. The special case `LF = 1` corresponds to the same internal logic used in the [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl) library.
  - Good accuracy and learning speed for small models. Achieves up to **98.56%** peak accuracy on **MNIST** and **89.67%** peak accuracy on **Fashion MNIST** using a model with **20 clauses** per class (10 positive + 10 negative). The original Tsetlin Machine from 2018 achieves approximately the same accuracy but with **2000** clauses per class.

The changes compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl) are located in the following functions: `check_clause()`, `feedback!()` and `train!()`.
Please, see the comments.

How to run MNIST example
------------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Run `julia --project=. -O3 -t 32 mnist.jl` where `32` is the number of your logical CPU cores.

How to run FashionMNIST example using convolutional preprocessing
-----------------------------------------------------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Run `julia --project=. -O3 -t 32 fmnist_conv.jl` where `32` is the number of your logical CPU cores.
