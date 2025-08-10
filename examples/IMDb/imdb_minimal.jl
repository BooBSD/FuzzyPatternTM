include("../../src/FuzzyPatternTMBinary.jl")

using Base.Threads
using .FuzzyPatternTM: TMInput, TMClassifier, train!, predict, accuracy, benchmark, load, save, initialize!


# Loading datasets
train = readlines("/tmp/IMDBTrainingData.txt")
test = readlines("/tmp/IMDBTestData.txt")

# Preparing datasets
x_train::Vector{TMInput} = Vector{TMInput}(undef, length(train))
y_train::Vector{Bool} = Vector{Bool}(undef, length(train))
@threads for i in eachindex(train)
    xy = [parse(Bool, x) for x in split(train[i], " ")]
    x_train[i] = TMInput(xy[1:length(xy) - 1])
    y_train[i] = xy[length(xy)]
end
x_test::Vector{TMInput} = Vector{TMInput}(undef, length(test))
y_test::Vector{Bool} = Vector{Bool}(undef, length(test))
@threads for i in eachindex(test)
    xy = [parse(Bool, x) for x in split(test[i], " ")]
    x_test[i] = TMInput(xy[1:length(xy) - 1])
    y_test[i] = xy[length(xy)]
end

CLAUSES = 1
T = 18
S = 1000
L = 64
LF = 64

const EPOCHS = 1000
const best_tms_size = 1

# Training the TM model
tm = TMClassifier{eltype(y_train)}(CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=220)
tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=best_tms_size, shuffle=true, batch=true, verbose=1, best_tms_compile=true)

# Saving model
save(tms[1][1], "/tmp/tm.tm")
# Loading model
tm = load("/tmp/tm.tm")
# Benchmark model
# 135 corresponds to a 5GB input dataset. Feel free to adjust this number if you like.
benchmark(tm, x_test, y_test, 135)
