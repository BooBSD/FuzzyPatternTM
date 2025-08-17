include("../../src/FuzzyPatternTM.jl")

try
    using MLDatasets: MNIST, FashionMNIST
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
end

using Printf: @printf
using MLDatasets: MNIST, FashionMNIST
using .FuzzyPatternTM: TMInput, TMClassifier, train!, predict, accuracy, save, load, unzip, booleanize, combine, optimize!, benchmark


x_train, y_train = unzip([MNIST(:train)...])
x_test, y_test = unzip([MNIST(:test)...])
# x_train, y_train = unzip([FashionMNIST(:train)...])
# x_test, y_test = unzip([FashionMNIST(:test)...])

# 4-bit booleanization
x_train = [booleanize(x, 0, 0.25, 0.5, 0.75) for x in x_train]
x_test = [booleanize(x, 0, 0.25, 0.5, 0.75) for x in x_test]

# Convert y_train and y_test to the Int8 type to save memory
y_train = Int8.(y_train)
y_test = Int8.(y_test)

CLAUSES = 20
T = 20
S = 200
L = 150
LF = 75

# CLAUSES = 200
# T = 20
# S = 200
# L = 16
# LF = 8

# CLAUSES = 512
# T = 32
# S = 200
# L = 16
# LF = 8

# CLAUSES = 2000
# T = 100
# S = 350
# L = 20
# LF = 10

# CLAUSES = 40
# T = 10
# S = 125
# L = 10
# LF = 5

EPOCHS = 2000
best_tms_size = 512

# Training the TM model
tm = TMClassifier{eltype(y_train)}(CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=200)  # include_limit=200 instead of 128 but you can try different numbers.
tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=best_tms_size, shuffle=true, batch=true, verbose=1)

save(tms, "/tmp/tms.tm")
tms = load("/tmp/tms.tm")

# Binomial combinatorial merge of trained TM models
tm, _ = combine(tms, 2, x_test, y_test, batch=true)
save(tm, "/tmp/tm2.tm")
tm = load("/tmp/tm2.tm")

# Optimizing the TM model
optimize!(tm, x_train)
save(tm, "/tmp/tm_optimized.tm")
tm_opt = load("/tmp/tm_optimized.tm")

benchmark(tm_opt, x_test, y_test, 5000, batch=true, warmup=true)
