include("FuzzyPatternsTM.jl")

try
    using MLDatasets: MNIST, FashionMNIST
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
end

using Printf: @printf
using MLDatasets: MNIST, FashionMNIST
using .FuzzyPatternsTM: TMInput, TMClassifier, train!, predict, accuracy, save, load, unzip, booleanize, combine, optimize!, benchmark


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

const CLAUSES = 20
const T = 16
const R = 0.992
const L = 150
const LF = 50

# const CLAUSES = 40
# const T = 10
# const R = 0.992
# const L = 10
# const LF = 5

# const CLAUSES = 2000
# const T = 100
# const R = 0.997
# const L = 20
# const LF = 10

const EPOCHS = 2000
const best_tms_size = 512

# Training the TM model
tm = TMClassifier{eltype(y_train)}(CLAUSES, T, R, L=L, LF=LF, states_num=256, include_limit=200)  # include_limit=200 instead of 128 but you can try different numbers.
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
