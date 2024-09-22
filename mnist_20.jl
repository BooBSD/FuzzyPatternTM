include("FuzzyPatternsTM.jl")

try
    using MLDatasets: MNIST, FashionMNIST
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
end

using Printf: @printf
using MLDatasets: MNIST, FashionMNIST
using .FuzzyPatternsTM: TMInput, TMClassifier, train!, predict, accuracy, save, load, unzip


x_train, y_train = unzip([MNIST(:train)...])
x_test, y_test = unzip([MNIST(:test)...])
# x_train, y_train = unzip([FashionMNIST(:train)...])
# x_test, y_test = unzip([FashionMNIST(:test)...])

# 4-bit booleanization
x_train = [TMInput(vec([
    [x > 0 ? true : false for x in i];
    [x > 0.25 ? true : false for x in i];
    [x > 0.5 ? true : false for x in i];
    [x > 0.75 ? true : false for x in i];
])) for i in x_train]
x_test = [TMInput(vec([
    [x > 0 ? true : false for x in i];
    [x > 0.25 ? true : false for x in i];
    [x > 0.5 ? true : false for x in i];
    [x > 0.75 ? true : false for x in i];
])) for i in x_test]

const EPOCHS = 2000
const CLAUSES = 20
const T = 16
const R = 0.992
const L = 150
const LF = 50

# Training the TM model
tm = TMClassifier(CLAUSES, T, R, L=L, LF=LF, states_num=256, include_limit=200)  # include_limit=200 instead of 128 but you can try different numbers.
# Batch inference is not implemented because of new algorithm.
best_tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, shuffle=true, verbose=1)

save(best_tms[1][2], "/tmp/tm_20.tm")
tm = load("/tmp/tm_20.tm")
