include("FuzzyPatternsTM.jl")
include("./utils.jl")

try
    using MLDatasets: MNIST, FashionMNIST
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
end

using Statistics
using MLDatasets: MNIST, FashionMNIST
using .FuzzyPatternsTM: TMInput, TMClassifier, train!, unzip


# x_train, y_train = unzip([MNIST(:train)...])
# x_test, y_test = unzip([MNIST(:test)...])
x_train, y_train = unzip([FashionMNIST(:train)...])
x_test, y_test = unzip([FashionMNIST(:test)...])

# Convolution kernels
Kx3 = [-1 0 1; -2 0 2; -1 0 1] * one(Float32)
#Kx5 = [-2 -1 0 1 2; -3 -2 0 2 3; -4 -3 0 3 4; -3 -2 0 2 3; -2 -1 0 1 2] * one(Float32)
Kx7 = [-3 -2 -1 0 1 2 3; -4 -3 -2 0 2 3 4; -5 -4 -3 0 3 4 5; -6 -5 -4 0 4 5 6; -5 -4 -3 0 3 4 5; -4 -3 -2 0 2 3 4; -3 -2 -1 0 1 2 3] * one(Float32)

#Kx3 = [0 1 2; -1 0 1; -2 -1 0] * one(Float32)
Kx5 = [0 1 2 3 4; -1 0 2 3 3; -2 -2 0 2 2; -3 -3 -2 0 1; -4 -3 -2 -1 0] * one(Float32)
#Kx7 = [0 1 2 3 4 5 6; -1 0 2 3 4 5 5; -2 -2 0 3 4 4 4; -3 -3 -3 0 3 3 3; -4 -4 -4 -3 0 2 2; -5 -5 -4 -3 -2 0 1; -6 -5 -4 -3 -2 -1 0] * one(Float32)

Ky3 = rotl90(Kx3)
Ky5 = rotl90(Kx5)
Ky7 = rotl90(Kx7)

Kp3 = 1  # Padding 1
Kp5 = 2  # Padding 2
Kp7 = 3  # Padding 3

x_train_conv_orient_x3 = [fastconv(x, Kx3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_train]
x_train_conv_orient_y3 = [fastconv(x, Ky3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_train]

x_test_conv_orient_x3 = [fastconv(x, Kx3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_test]
x_test_conv_orient_y3 = [fastconv(x, Ky3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_test]

x_train_conv_orient_x5 = [fastconv(x, Kx5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_train]
x_train_conv_orient_y5 = [fastconv(x, Ky5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_train]

x_test_conv_orient_x5 = [fastconv(x, Kx5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_test]
x_test_conv_orient_y5 = [fastconv(x, Ky5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_test]

x_train_conv_orient_x7 = [fastconv(x, Kx7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_train]
x_train_conv_orient_y7 = [fastconv(x, Ky7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_train]

x_test_conv_orient_x7 = [fastconv(x, Kx7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_test]
x_test_conv_orient_y7 = [fastconv(x, Ky7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_test]

train_hist_data = vec(vcat(x_train...))

x3_hist_data = vec(vcat(x_train_conv_orient_x3...))
y3_hist_data = vec(vcat(x_train_conv_orient_y3...))

x5_hist_data = vec(vcat(x_train_conv_orient_x5...))
y5_hist_data = vec(vcat(x_train_conv_orient_y5...))

x7_hist_data = vec(vcat(x_train_conv_orient_x7...))
y7_hist_data = vec(vcat(x_train_conv_orient_y7...))

train_hist_25::Float64 = quantile([x for x in train_hist_data if x > 0], 0.25)
train_hist_50::Float64 = quantile([x for x in train_hist_data if x > 0], 0.50)
train_hist_75::Float64= quantile([x for x in train_hist_data if x > 0], 0.75)

println(train_hist_25)
println(train_hist_50)
println(train_hist_75)

x3_hist_pos_25::Float64 = quantile([x for x in x3_hist_data if x > 0], 0.25)
    x3_hist_pos_34::Float64 = quantile([x for x in x3_hist_data if x > 0], 0.34)
x3_hist_pos_50::Float64 = quantile([x for x in x3_hist_data if x > 0], 0.50)
x3_hist_pos_75::Float64 = quantile([x for x in x3_hist_data if x > 0], 0.75)
x3_hist_neg_25::Float64 = quantile([x for x in x3_hist_data if x < 0], 1 - 0.25)
    x3_hist_neg_34::Float64 = quantile([x for x in x3_hist_data if x < 0], 1 - 0.34)
x3_hist_neg_50::Float64 = quantile([x for x in x3_hist_data if x < 0], 1 - 0.50)
x3_hist_neg_75::Float64 = quantile([x for x in x3_hist_data if x < 0], 1 - 0.75)

y3_hist_pos_25::Float64 = quantile([x for x in y3_hist_data if x > 0], 0.25)
    y3_hist_pos_34::Float64 = quantile([x for x in y3_hist_data if x > 0], 0.34)
y3_hist_pos_50::Float64 = quantile([x for x in y3_hist_data if x > 0], 0.50)
y3_hist_pos_75::Float64 = quantile([x for x in y3_hist_data if x > 0], 0.75)
y3_hist_neg_25::Float64 = quantile([x for x in y3_hist_data if x < 0], 1 - 0.25)
    y3_hist_neg_34::Float64 = quantile([x for x in y3_hist_data if x < 0], 1 - 0.34)
y3_hist_neg_50::Float64 = quantile([x for x in y3_hist_data if x < 0], 1 - 0.50)
y3_hist_neg_75::Float64 = quantile([x for x in y3_hist_data if x < 0], 1 - 0.75)

x5_hist_pos_25::Float64 = quantile([x for x in x5_hist_data if x > 0], 0.25)
    x5_hist_pos_34::Float64 = quantile([x for x in x5_hist_data if x > 0], 0.34)
x5_hist_pos_50::Float64 = quantile([x for x in x5_hist_data if x > 0], 0.50)
x5_hist_pos_75::Float64 = quantile([x for x in x5_hist_data if x > 0], 0.75)
x5_hist_neg_25::Float64 = quantile([x for x in x5_hist_data if x < 0], 1 - 0.25)
    x5_hist_neg_34::Float64 = quantile([x for x in x5_hist_data if x < 0], 1 - 0.34)
x5_hist_neg_50::Float64 = quantile([x for x in x5_hist_data if x < 0], 1 - 0.50)
x5_hist_neg_75::Float64 = quantile([x for x in x5_hist_data if x < 0], 1 - 0.75)

y5_hist_pos_25::Float64 = quantile([x for x in y5_hist_data if x > 0], 0.25)
    y5_hist_pos_34::Float64 = quantile([x for x in y5_hist_data if x > 0], 0.34)
y5_hist_pos_50::Float64 = quantile([x for x in y5_hist_data if x > 0], 0.50)
y5_hist_pos_75::Float64 = quantile([x for x in y5_hist_data if x > 0], 0.75)
y5_hist_neg_25::Float64 = quantile([x for x in y5_hist_data if x < 0], 1 - 0.25)
    y5_hist_neg_34::Float64 = quantile([x for x in y5_hist_data if x < 0], 1 - 0.34)
y5_hist_neg_50::Float64 = quantile([x for x in y5_hist_data if x < 0], 1 - 0.50)
y5_hist_neg_75::Float64 = quantile([x for x in y5_hist_data if x < 0], 1 - 0.75)

x7_hist_pos_25::Float64 = quantile([x for x in x7_hist_data if x > 0], 0.25)
    x7_hist_pos_34::Float64 = quantile([x for x in x7_hist_data if x > 0], 0.34)
x7_hist_pos_50::Float64 = quantile([x for x in x7_hist_data if x > 0], 0.50)
x7_hist_pos_75::Float64 = quantile([x for x in x7_hist_data if x > 0], 0.75)
x7_hist_neg_25::Float64 = quantile([x for x in x7_hist_data if x < 0], 1 - 0.25)
    x7_hist_neg_34::Float64 = quantile([x for x in x7_hist_data if x < 0], 1 - 0.34)
x7_hist_neg_50::Float64 = quantile([x for x in x7_hist_data if x < 0], 1 - 0.50)
x7_hist_neg_75::Float64 = quantile([x for x in x7_hist_data if x < 0], 1 - 0.75)

y7_hist_pos_25::Float64 = quantile([x for x in y7_hist_data if x > 0], 0.25)
    y7_hist_pos_34::Float64 = quantile([x for x in y7_hist_data if x > 0], 0.34)
y7_hist_pos_50::Float64 = quantile([x for x in y7_hist_data if x > 0], 0.50)
y7_hist_pos_75::Float64 = quantile([x for x in y7_hist_data if x > 0], 0.75)
y7_hist_neg_25::Float64 = quantile([x for x in y7_hist_data if x < 0], 1 - 0.25)
    y7_hist_neg_34::Float64 = quantile([x for x in y7_hist_data if x < 0], 1 - 0.34)
y7_hist_neg_50::Float64 = quantile([x for x in y7_hist_data if x < 0], 1 - 0.50)
y7_hist_neg_75::Float64 = quantile([x for x in y7_hist_data if x < 0], 1 - 0.75)

# Booleanization
function bools(x7, y7, x5, y5, x3, y3, io)
    return TMInput([
        # Raw pixels
        [x > 0 ? true : false for x in io];
        [x > train_hist_25 ? true : false for x in io];
        [x > train_hist_50 ? true : false for x in io];
        [x > train_hist_75 ? true : false for x in io];

        # 3x3 convolution results
        [x > 0 ? true : false for x in x3];
        [x > x3_hist_pos_25 ? true : false for x in x3];
            [x > x3_hist_pos_34 ? true : false for x in x3];
        [x > x3_hist_pos_50 ? true : false for x in x3];
        [x > x3_hist_pos_75 ? true : false for x in x3];
        # [x < 0 ? true : false for x in x3];
        [x < x3_hist_neg_25 ? true : false for x in x3];
            [x < x3_hist_neg_34 ? true : false for x in x3];
        [x < x3_hist_neg_50 ? true : false for x in x3];
        [x < x3_hist_neg_75 ? true : false for x in x3];

        [x > 0 ? true : false for x in y3];
        [x > y3_hist_pos_25 ? true : false for x in y3];
            [x > y3_hist_pos_34 ? true : false for x in y3];
        [x > y3_hist_pos_50 ? true : false for x in y3];
        [x > y3_hist_pos_75 ? true : false for x in y3];
        # [x < 0 ? true : false for x in y3];
        [x < y3_hist_neg_25 ? true : false for x in y3];
            [x < y3_hist_neg_34 ? true : false for x in y3];
        [x < y3_hist_neg_50 ? true : false for x in y3];
        [x < y3_hist_neg_75 ? true : false for x in y3];

        # 5x5 convolution results
        [x > 0 ? true : false for x in x5];
        [x > x5_hist_pos_25 ? true : false for x in x5];
            [x > x5_hist_pos_34 ? true : false for x in x5];
        [x > x5_hist_pos_50 ? true : false for x in x5];
        [x > x5_hist_pos_75 ? true : false for x in x5];
        # [x < 0 ? true : false for x in x5];
        [x < x5_hist_neg_25 ? true : false for x in x5];
            [x < x5_hist_neg_34 ? true : false for x in x5];
        [x < x5_hist_neg_50 ? true : false for x in x5];
        [x < x5_hist_neg_75 ? true : false for x in x5];

        [x > 0 ? true : false for x in y5];
        [x > y5_hist_pos_25 ? true : false for x in y5];
            [x > y5_hist_pos_34 ? true : false for x in y5];
        [x > y5_hist_pos_50 ? true : false for x in y5];
        [x > y5_hist_pos_75 ? true : false for x in y5];
        # [x < 0 ? true : false for x in y5];
        [x < y5_hist_neg_25 ? true : false for x in y5];
            [x < y5_hist_neg_34 ? true : false for x in y5];
        [x < y5_hist_neg_50 ? true : false for x in y5];
        [x < y5_hist_neg_75 ? true : false for x in y5];

        # 7x7 convolution results
        [x > 0 ? true : false for x in x7];
        [x > x7_hist_pos_25 ? true : false for x in x7];
            [x > x7_hist_pos_34 ? true : false for x in x7];
        [x > x7_hist_pos_50 ? true : false for x in x7];
        [x > x7_hist_pos_75 ? true : false for x in x7];
        # [x < 0 ? true : false for x in x7];
        [x < x7_hist_neg_25 ? true : false for x in x7];
            [x < x7_hist_neg_34 ? true : false for x in x7];
        [x < x7_hist_neg_50 ? true : false for x in x7];
        [x < x7_hist_neg_75 ? true : false for x in x7];

        [x > 0 ? true : false for x in y7];
        [x > y7_hist_pos_25 ? true : false for x in y7];
            [x > y7_hist_pos_34 ? true : false for x in y7];
        [x > y7_hist_pos_50 ? true : false for x in y7];
        [x > y7_hist_pos_75 ? true : false for x in y7];
        # [x < 0 ? true : false for x in y7];
        [x < y7_hist_neg_25 ? true : false for x in y7];
            [x < y7_hist_neg_34 ? true : false for x in y7];
        [x < y7_hist_neg_50 ? true : false for x in y7];
        [x < y7_hist_neg_75 ? true : false for x in y7];
    ])
end

x_train = [bools(x7, y7, x5, y5, x3, y3, io) for (x7, y7, x5, y5, x3, y3, io) in zip(x_train_conv_orient_x7, x_train_conv_orient_y7, x_train_conv_orient_x5, x_train_conv_orient_y5, x_train_conv_orient_x3, x_train_conv_orient_y3, x_train)]
x_test = [bools(x7, y7, x5, y5, x3, y3, io) for (x7, y7, x5, y5, x3, y3, io) in zip(x_test_conv_orient_x7, x_test_conv_orient_y7, x_test_conv_orient_x5, x_test_conv_orient_y5, x_test_conv_orient_x3, x_test_conv_orient_y3, x_test)]

y_train = Int8.(y_train)
y_test = Int8.(y_test)

# CLAUSES = 2000  # Best accuracy: 94.10% after 160 epoch
# T = 350
# S = 1000
# L = 90
# LF = 30

CLAUSES = 20  # Best accuracy: 93.32% after 150 epochs
T = 90
S = 1000
L = 400
LF = 200

# CLAUSES = 2  # Best accuracy: 92.20% after 900 epochs
# T = 56
# S = 1000
# L = 1000
# LF = 800

EPOCHS = 2000

# Training the TM model
tm = TMClassifier{eltype(y_train)}(CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=220)
m_best, _ = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, shuffle=true, verbose=1)
