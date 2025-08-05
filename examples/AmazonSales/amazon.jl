include("../../src/FuzzyPatternTM.jl")

using .FuzzyPatternTM: TMInput, TMClassifier, train!


X_train = readlines("/tmp/Amazon_X_train.txt")
Y_train = readlines("/tmp/Amazon_Y_train.txt")
X_test = readlines("/tmp/Amazon_X_test.txt")
Y_test = readlines("/tmp/Amazon_Y_test.txt")

x_train = [TMInput([parse(Bool, x) for x in split(X, " ")]) for X in X_train]
x_test = [TMInput([parse(Bool, x) for x in split(X, " ")]) for X in X_test]
y_train = [parse(Int8, Y) for Y in Y_train]
y_test = [parse(Int8, Y) for Y in Y_test]

CLAUSES = 2000
T = 100
S = 1000
L = 50
LF = 10

EPOCHS = 1000
best_tms_size = 1

# Training the TM model
tm = TMClassifier{eltype(y_train)}(CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=200)
tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=best_tms_size, shuffle=true, batch=true, verbose=1)
