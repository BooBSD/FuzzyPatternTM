# Fuzzy-Pattern Tsetlin Machine

## Abstract

The "*all-or-nothing*" clause evaluation strategy is a core mechanism in the Tsetlin Machine (TM) family of algorithms. In this approach, each clause—a logical pattern composed of binary literals mapped to input data—is disqualified from voting if even a single literal fails. Due to this strict requirement, standard TMs must employ thousands of clauses to achieve competitive accuracy.

This paper introduces the **Fuzzy-Pattern Tsetlin Machine** (FPTM), a novel variant where clause evaluation is fuzzy rather than strict. If some literals in a clause fail, the remaining ones can still contribute to the overall vote with a proportionally reduced score. As a result, each clause effectively consists of sub-patterns that adapt individually to the input, enabling more flexible, efficient, and robust pattern matching.

The proposed fuzzy mechanism significantly reduces the number of required clauses, memory footprint, and training time, while simultaneously enhancing accuracy. Experiments on the *IMDb* dataset demonstrate a median peak test accuracy of **90.15%** using only **one clause per class**, representing a **50× reduction** in both clause count and memory usage compared to Weighted CoalescedTM, a leading TM variant. The Tsetlin Automata state matrix fits within **50 KB**, enabling online training on modern microcontrollers. Training is **36× faster** than CoalescedTM on a single-threaded CPU, and **316× faster** with 32-thread parallelism. For instance, while CoalescedTM requires nearly **4 hours** to train for 1000 epochs, FPTM completes the same task in just **45 seconds** with comparable accuracy. Inference throughput reaches up to **34.5 million predictions per second** in batch mode, corresponding to **51.4 GB/s** on a desktop CPU.

Additional experiments on the *Fashion-MNIST* dataset with convolutional preprocessing yield similar results: **92.18%** test accuracy with only **2 clauses per class**, **93.19%** with 20 clauses per class, and **94.68%** with 8000 clauses per class. This demonstrates an approximate **400× reduction** in clause count compared to previous Composite TM approach, whose best performance was **93.00%** accuracy using a model with **8000** clauses per class. Finally, on the noisy *Amazon Sales* dataset (**20% noise**), FPTM achieves **85.22%** test accuracy, outperforming GraphTM (78.17%) and GraphCNN (66.23%).

Here is the tiny **20-clause** model training result for the **MNIST** dataset:
<img width="698" alt="Experimental Fuzzy Patterns Tsetlin Machine MNIST accuracy 98.56%" src="https://github.com/user-attachments/assets/05768a26-036a-40ce-b548-95925e96a01d">

## Key features compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl)

  - New hyperparameter `LF` that sets the number of literal misses allowed for the clause. The special case `LF = 1` corresponds to the same internal logic used in the [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl) library.
  - Good accuracy and learning speed for small models. Achieves up to **98.56%** peak accuracy on **MNIST** and **89.67%** peak accuracy on **Fashion MNIST** using a model with **20 clauses** per class (10 positive + 10 negative). The original Tsetlin Machine from 2018 achieves approximately the same accuracy but with **2000** clauses per class.

The changes compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl) are located in the following functions: `check_clause()`, `feedback!()` and `train!()`.
Please, see the comments.

## How to Run Examples

- Ensure that you have the latest version of the [Julia](https://julialang.org/downloads/) language installed.
- Some examples require dataset preparation scripts written in [Python](https://www.python.org/downloads/). To install the necessary dependencies, run:
```shell
pip install -r requirements.txt
```

### IMDb Example

Prepare IMDb dataset:

```shell
python examples/IMDb/prepare_dataset.py
```

Run the IMDb training and benchmarking example:

```shell
julia --project=. -O3 -t 32 examples/IMDb/imdb.jl
```
Here, `32` should be replaced with the number of logical CPU cores on your machine.

### Noisy Amazon Sales Example

Prepare noisy Amazon Sales dataset:

```shell
python examples/AmazonSales/prepare_dataset.py --dataset_noise_ratio=0.005
```

Run the Noisy Amazon Sales training example:

```shell
julia --project=. -O3 -t 32 examples/AmazonSales/amazon.jl
```
Here, `32` should be replaced with the number of logical CPU cores on your machine.

### Fashion-MNIST Example Using Convolutional Preprocessing

Run the Fashion-MNIST training example:

```shell
julia --project=. -O3 -t 32 examples/FashionMNIST/fmnist_conv.jl
```
Here, `32` should be replaced with the number of logical CPU cores on your machine.

### Fashion-MNIST Example Using Convolutional Preprocessing and Data Augmentation

To achieve maximum test accuracy, prepare the Fashion-MNIST dataset with data augmentation:

```shell
julia --project=. -O3 -t 32 examples/FashionMNIST/prepare_augmented_dataset.jl
```

Run the large model training example on Fashion-MNIST:

```shell
julia --project=. -O3 -t 32 examples/FashionMNIST/fmnist_conv_augmented.jl
```
Here, `32` should be replaced with the number of logical CPU cores on your machine.

### MNIST Example

Run the MNIST training example:

```shell
julia --project=. -O3 -t 32 examples/MNIST/mnist.jl
```
Here, `32` should be replaced with the number of logical CPU cores on your machine.
