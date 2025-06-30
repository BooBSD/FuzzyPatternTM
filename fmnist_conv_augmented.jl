include("FuzzyPatternTM.jl")
include("./utils.jl")

try
    using MLDatasets: FashionMNIST
    using Augmentor
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
    Pkg.add("Augmentor")
end

using Base.Threads
using Statistics
using Serialization
using Augmentor
using MLDatasets: FashionMNIST
using .FuzzyPatternTM: TMInput, TMClassifier, train!, unzip, save, load, combine, accuracy, predict


x_train, y_train = unzip([FashionMNIST(:train)...])
x_test, y_test = unzip([FashionMNIST(:test)...])

x_train_aug1 = deepcopy(x_train)
x_train_aug2 = deepcopy(x_train)
x_train_aug3 = deepcopy(x_train)
x_train_aug4 = deepcopy(x_train)
x_train_aug5 = deepcopy(x_train)
# x_train_aug6 = deepcopy(x_train)
# x_train_aug7 = deepcopy(x_train)
# x_train_aug8 = deepcopy(x_train)
# x_train_aug9 = deepcopy(x_train)
# x_test = deepcopy(x_train)
# y_test = deepcopy(y_train)

# dist = ElasticDistortion(6, scale=0.3, border=true) |>
# Rotate([10, -5, -3, 0, 3, 5, 10]) |>
# ShearX(-3:3) * ShearY(-3:3) |>
# CropSize(28, 28) |>
# Zoom(0.9:0.1:1.2)

# augmentbatch!(x_train_aug1, x_train, Zoom(0.915))    # 26 px
# augmentbatch!(x_train_aug2, x_train, Zoom(1.085))   # 30 px
# augmentbatch!(x_train_aug3, x_train, Rotate(-7.5) |> CropSize(28, 28))
# augmentbatch!(x_train_aug4, x_train, Rotate(7.5) |> CropSize(28, 28))
augmentbatch!(x_train_aug1, x_train, Zoom(0.9))
augmentbatch!(x_train_aug2, x_train, Zoom(1.1))
augmentbatch!(x_train_aug3, x_train, Rotate(-6) |> CropSize(28, 28))
augmentbatch!(x_train_aug4, x_train, Rotate(6) |> CropSize(28, 28))
augmentbatch!(x_train_aug5, x_train, FlipY())  # Yes, Y, because of dataset orientation
# augmentbatch!(x_train_aug6, x_train, FlipY() |> Zoom(0.915))
# augmentbatch!(x_train_aug7, x_train, FlipY() |> Zoom(1.085))
# augmentbatch!(x_train_aug8, x_train, FlipY() |> Rotate(-7.5) |> CropSize(28, 28))
# augmentbatch!(x_train_aug9, x_train, FlipY() |> Rotate(7.5) |> CropSize(28, 28))
#augmentbatch!(x_test, x_train, Zoom(1.143))    # 32 px  ??? %
#augmentbatch!(x_train_aug, x_train, ElasticDistortion(10, 10, 0.2, 3, 2, true))

x_train = [x_train; x_train_aug1; x_train_aug2; x_train_aug3; x_train_aug4; x_train_aug5]
y_train = [y_train; y_train; y_train; y_train; y_train; y_train]
# x_train = [x_train; x_train_aug1; x_train_aug2; x_train_aug3; x_train_aug4; x_train_aug5; x_train_aug6; x_train_aug7; x_train_aug8; x_train_aug9]
# y_train = [y_train; y_train; y_train; y_train; y_train; y_train; y_train; y_train; y_train; y_train]

print("Preparing input data... ")

# Convolution kernels
Kx3 = [-1 0 1; -2 0 2; -1 0 1] * one(Float32)
Kx5 = [-2 -1 0 1 2; -3 -2 0 2 3; -4 -3 0 3 4; -3 -2 0 2 3; -2 -1 0 1 2] * one(Float32)  # Comment for CLAUSES = 2
Kx7 = [-3 -2 -1 0 1 2 3; -4 -3 -2 0 2 3 4; -5 -4 -3 0 3 4 5; -6 -5 -4 0 4 5 6; -5 -4 -3 0 3 4 5; -4 -3 -2 0 2 3 4; -3 -2 -1 0 1 2 3] * one(Float32)
#Kx9 = [-4 -3 -2 -1 0 1 2 3 4; -5 -4 -3 -2 0 2 3 4 5; -6 -5 -4 -3 0 3 4 5 6; -7 -6 -5 -4 0 4 5 6 7; -8 -7 -6 -5 0 5 6 7 8; -7 -6 -5 -4 0 4 5 6 7; -6 -5 -4 -3 0 3 4 5 6; -5 -4 -3 -2 0 2 3 4 5; -4 -3 -2 -1 0 1 2 3 4] * one(Float32)

# Sobel kernels
# Kx3 = [-1 -2 -1; 0 0 0; 1 2 1] * one(Float32)
# Kx5 = [-1 -2 -4 -2 -1; -2 -4 -8 -4 -2; 0 0 0 0 0; 2 4 8 4 2; 1 2 4 2 1] * one(Float32)
# Kx7 = [-1 -2 -4 -8 -4 -2 -1; -2 -4 -8 -16 -8 -4 -2; -4 -8 -16 -32 -16 -8 -4; 0 0 0 0 0 0 0; 4 8 16 32 16 8 4; 2 4 8 16 8 4 2; 1 2 4 8 4 2 1] * one(Float32)

#Kx3 = [0 1 2; -1 0 1; -2 -1 0] * one(Float32)
#Kx5 = [0 1 2 3 4; -1 0 2 3 3; -2 -2 0 2 2; -3 -3 -2 0 1; -4 -3 -2 -1 0] * one(Float32)  # Uncomment for CLAUSES = 2
#Kx7 = [0 1 2 3 4 5 6; -1 0 2 3 4 5 5; -2 -2 0 3 4 4 4; -3 -3 -3 0 3 3 3; -4 -4 -4 -3 0 2 2; -5 -5 -4 -3 -2 0 1; -6 -5 -4 -3 -2 -1 0] * one(Float32)
#Kx9 = [0 1 2 3 4 5 6 7 8; -1 0 2 3 4 5 6 7 7; -2 -2 0 3 4 5 6 6 6; -3 -3 -3 0 4 5 5 5 5; -4 -4 -4 -4 0 4 4 4 4; -5 -5 -5 -5 -4 0 3 3 3; -6 -6 -6 -5 -4 -3 0 2 2; -7 -7 -6 -5 -4 -3 -2 0 1; -8 -7 -6 -5 -4 -3 -2 -1 0] * one(Float32)

#Kx3 = [-1 -1 -1; 0 0 0; 1 1 1] * one(Float32)
#Kx7 = [-3 -3 -3 -3 -3; -10 -10 -10 -10 -10; 0 0 0 0 0; 10 10 10 10 10; 3 3 3 3 3] * one(Float32)
#Kx7 = [-1 -1 -1 -1 -1 -1 -1; -1 -1 -1 -1 -1 -1 -1; -1 -1 -1 -1 -1 -1 -1; 0 0 0 0 0 0 0; 1 1 1 1 1 1 1; 1 1 1 1 1 1 1; 1 1 1 1 1 1 1] * one(Float32)

Kx9 = [-1 -1 -1; 2 2 2; -1 -1 -1] * one(Float32)
#Kx9 = [-4 -4 -4 -4 -4; -1 -1 -1 -1 -1; 2 2 2 2 2; 1 1 1 1 1; 4 4 4 4 4] * one(Float32)
#Kx9 = [-2 -1 0; -1 1 1; 0 1 2] * one(Float32)
#Kx9 = [0 0 -1 -1 -1 0 0; 0 -1 0 0 0 -1 0; -1 0 1 1 1 0 -1; -1 0 1 2 1 0 -1; -1 0 1 1 1 0 -1; 0 -1 0 0 0 -1 0; 0 0 -1 -1 -1 0 0] * one(Float32)

# Kx3 = [-1 -1 -1; 0 0 0; 1 1 1] * one(Float32)
# Kx5 = [-2 -2 -2 -2 -2; -1 -1 -1 -1 -1; 0 0 0 0 0; 1 1 1 1 1; 2 2 2 2 2] * one(Float32)  # Comment for CLAUSES = 2
# Kx7 = [-3 -3 -3 -3 -3 -3 -3; -2 -2 -2 -2 -2 -2 -2; -1 -1 -1 -1 -1 -1 -1; 0 0 0 0 0 0 0; 1 1 1 1 1 1 1; 2 2 2 2 2 2 2; 3 3 3 3 3 3 3] * one(Float32)

# Gabor
#Kx9::Matrix{Float32} = [-0.031 -0.063 -0.075 -0.063 -0.031; -0.063 -0.125 -0.150 -0.125 -0.063; 0.0 0.0 0.6 0.0 0.0; 0.063 0.125 0.150 0.125 0.063; 0.031 0.063 0.075 0.063 0.031]


Ky3 = rotl90(Kx3)
Ky5 = rotl90(Kx5)
Ky7 = rotl90(Kx7)
Ky9 = rotl90(Kx9)

Kp3 = 1  # Padding 1
Kp5 = 2  # Padding 2
Kp7 = 3  # Padding 3
Kp9 = 1  # Padding 4

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

x_train_conv_orient_x9 = [fastconv(x, Kx9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_train]
x_train_conv_orient_y9 = [fastconv(x, Ky9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_train]

x_test_conv_orient_x9 = [fastconv(x, Kx9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_test]
x_test_conv_orient_y9 = [fastconv(x, Ky9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_test]


train_hist_data = vec(vcat(x_train...))

x3_hist_data = vec(vcat(x_train_conv_orient_x3...))
y3_hist_data = vec(vcat(x_train_conv_orient_y3...))

x5_hist_data = vec(vcat(x_train_conv_orient_x5...))
y5_hist_data = vec(vcat(x_train_conv_orient_y5...))

x7_hist_data = vec(vcat(x_train_conv_orient_x7...))
y7_hist_data = vec(vcat(x_train_conv_orient_y7...))

x9_hist_data = vec(vcat(x_train_conv_orient_x9...))
y9_hist_data = vec(vcat(x_train_conv_orient_y9...))

raw_hist_25::Float64 = quantile([x for x in train_hist_data if x > 0], 0.25)
raw_hist_50::Float64 = quantile([x for x in train_hist_data if x > 0], 0.50)
raw_hist_75::Float64= quantile([x for x in train_hist_data if x > 0], 0.75)

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

x9_hist_pos_25::Float64 = quantile([x for x in x9_hist_data if x > 0], 0.25)
x9_hist_pos_34::Float64 = quantile([x for x in x9_hist_data if x > 0], 0.34)
x9_hist_pos_50::Float64 = quantile([x for x in x9_hist_data if x > 0], 0.50)
x9_hist_pos_75::Float64 = quantile([x for x in x9_hist_data if x > 0], 0.75)
x9_hist_neg_25::Float64 = quantile([x for x in x9_hist_data if x < 0], 1 - 0.25)
x9_hist_neg_34::Float64 = quantile([x for x in x9_hist_data if x < 0], 1 - 0.34)
x9_hist_neg_50::Float64 = quantile([x for x in x9_hist_data if x < 0], 1 - 0.50)
x9_hist_neg_75::Float64 = quantile([x for x in x9_hist_data if x < 0], 1 - 0.75)

y9_hist_pos_25::Float64 = quantile([x for x in y9_hist_data if x > 0], 0.25)
y9_hist_pos_34::Float64 = quantile([x for x in y9_hist_data if x > 0], 0.34)
y9_hist_pos_50::Float64 = quantile([x for x in y9_hist_data if x > 0], 0.50)
y9_hist_pos_75::Float64 = quantile([x for x in y9_hist_data if x > 0], 0.75)
y9_hist_neg_25::Float64 = quantile([x for x in y9_hist_data if x < 0], 1 - 0.25)
y9_hist_neg_34::Float64 = quantile([x for x in y9_hist_data if x < 0], 1 - 0.34)
y9_hist_neg_50::Float64 = quantile([x for x in y9_hist_data if x < 0], 1 - 0.50)
y9_hist_neg_75::Float64 = quantile([x for x in y9_hist_data if x < 0], 1 - 0.75)


# Booleanization
function bools(raw, x3, y3, x5, y5, x7, y7, x9, y9)
    # raw_hist_25::Float64 = quantile([x for x in raw if x > 0], 0.25)
    # raw_hist_50::Float64 = quantile([x for x in raw if x > 0], 0.50)
    # raw_hist_75::Float64= quantile([x for x in raw if x > 0], 0.75)

    # x3_hist_pos_25::Float64 = quantile([x for x in x3 if x > 0], 0.25)
    # x3_hist_pos_34::Float64 = quantile([x for x in x3 if x > 0], 0.34)
    # x3_hist_pos_50::Float64 = quantile([x for x in x3 if x > 0], 0.50)
    # x3_hist_pos_75::Float64 = quantile([x for x in x3 if x > 0], 0.75)
    # x3_hist_neg_25::Float64 = quantile([x for x in x3 if x < 0], 1 - 0.25)
    # x3_hist_neg_34::Float64 = quantile([x for x in x3 if x < 0], 1 - 0.34)
    # x3_hist_neg_50::Float64 = quantile([x for x in x3 if x < 0], 1 - 0.50)
    # x3_hist_neg_75::Float64 = quantile([x for x in x3 if x < 0], 1 - 0.75)

    # y3_hist_pos_25::Float64 = quantile([x for x in y3 if x > 0], 0.25)
    # y3_hist_pos_34::Float64 = quantile([x for x in y3 if x > 0], 0.34)
    # y3_hist_pos_50::Float64 = quantile([x for x in y3 if x > 0], 0.50)
    # y3_hist_pos_75::Float64 = quantile([x for x in y3 if x > 0], 0.75)
    # y3_hist_neg_25::Float64 = quantile([x for x in y3 if x < 0], 1 - 0.25)
    # y3_hist_neg_34::Float64 = quantile([x for x in y3 if x < 0], 1 - 0.34)
    # y3_hist_neg_50::Float64 = quantile([x for x in y3 if x < 0], 1 - 0.50)
    # y3_hist_neg_75::Float64 = quantile([x for x in y3 if x < 0], 1 - 0.75)

    # x5_hist_pos_25::Float64 = quantile([x for x in x5 if x > 0], 0.25)
    # x5_hist_pos_34::Float64 = quantile([x for x in x5 if x > 0], 0.34)
    # x5_hist_pos_50::Float64 = quantile([x for x in x5 if x > 0], 0.50)
    # x5_hist_pos_75::Float64 = quantile([x for x in x5 if x > 0], 0.75)
    # x5_hist_neg_25::Float64 = quantile([x for x in x5 if x < 0], 1 - 0.25)
    # x5_hist_neg_34::Float64 = quantile([x for x in x5 if x < 0], 1 - 0.34)
    # x5_hist_neg_50::Float64 = quantile([x for x in x5 if x < 0], 1 - 0.50)
    # x5_hist_neg_75::Float64 = quantile([x for x in x5 if x < 0], 1 - 0.75)

    # y5_hist_pos_25::Float64 = quantile([x for x in y5 if x > 0], 0.25)
    # y5_hist_pos_34::Float64 = quantile([x for x in y5 if x > 0], 0.34)
    # y5_hist_pos_50::Float64 = quantile([x for x in y5 if x > 0], 0.50)
    # y5_hist_pos_75::Float64 = quantile([x for x in y5 if x > 0], 0.75)
    # y5_hist_neg_25::Float64 = quantile([x for x in y5 if x < 0], 1 - 0.25)
    # y5_hist_neg_34::Float64 = quantile([x for x in y5 if x < 0], 1 - 0.34)
    # y5_hist_neg_50::Float64 = quantile([x for x in y5 if x < 0], 1 - 0.50)
    # y5_hist_neg_75::Float64 = quantile([x for x in y5 if x < 0], 1 - 0.75)

    # x7_hist_pos_25::Float64 = quantile([x for x in x7 if x > 0], 0.25)
    # x7_hist_pos_34::Float64 = quantile([x for x in x7 if x > 0], 0.34)
    # x7_hist_pos_50::Float64 = quantile([x for x in x7 if x > 0], 0.50)
    # x7_hist_pos_75::Float64 = quantile([x for x in x7 if x > 0], 0.75)
    # x7_hist_neg_25::Float64 = quantile([x for x in x7 if x < 0], 1 - 0.25)
    # x7_hist_neg_34::Float64 = quantile([x for x in x7 if x < 0], 1 - 0.34)
    # x7_hist_neg_50::Float64 = quantile([x for x in x7 if x < 0], 1 - 0.50)
    # x7_hist_neg_75::Float64 = quantile([x for x in x7 if x < 0], 1 - 0.75)

    # y7_hist_pos_25::Float64 = quantile([x for x in y7 if x > 0], 0.25)
    # y7_hist_pos_34::Float64 = quantile([x for x in y7 if x > 0], 0.34)
    # y7_hist_pos_50::Float64 = quantile([x for x in y7 if x > 0], 0.50)
    # y7_hist_pos_75::Float64 = quantile([x for x in y7 if x > 0], 0.75)
    # y7_hist_neg_25::Float64 = quantile([x for x in y7 if x < 0], 1 - 0.25)
    # y7_hist_neg_34::Float64 = quantile([x for x in y7 if x < 0], 1 - 0.34)
    # y7_hist_neg_50::Float64 = quantile([x for x in y7 if x < 0], 1 - 0.50)
    # y7_hist_neg_75::Float64 = quantile([x for x in y7 if x < 0], 1 - 0.75)

    # x9_hist_pos_25::Float64 = quantile([x for x in x9 if x > 0], 0.25)
    # x9_hist_pos_34::Float64 = quantile([x for x in x9 if x > 0], 0.34)
    # x9_hist_pos_50::Float64 = quantile([x for x in x9 if x > 0], 0.50)
    # x9_hist_pos_75::Float64 = quantile([x for x in x9 if x > 0], 0.75)
    # x9_hist_neg_25::Float64 = quantile([x for x in x9 if x < 0], 1 - 0.25)
    # x9_hist_neg_34::Float64 = quantile([x for x in x9 if x < 0], 1 - 0.34)
    # x9_hist_neg_50::Float64 = quantile([x for x in x9 if x < 0], 1 - 0.50)
    # x9_hist_neg_75::Float64 = quantile([x for x in x9 if x < 0], 1 - 0.75)

    # y9_hist_pos_25::Float64 = quantile([x for x in y9 if x > 0], 0.25)
    # y9_hist_pos_34::Float64 = quantile([x for x in y9 if x > 0], 0.34)
    # y9_hist_pos_50::Float64 = quantile([x for x in y9 if x > 0], 0.50)
    # y9_hist_pos_75::Float64 = quantile([x for x in y9 if x > 0], 0.75)
    # y9_hist_neg_25::Float64 = quantile([x for x in y9 if x < 0], 1 - 0.25)
    # y9_hist_neg_34::Float64 = quantile([x for x in y9 if x < 0], 1 - 0.34)
    # y9_hist_neg_50::Float64 = quantile([x for x in y9 if x < 0], 1 - 0.50)
    # y9_hist_neg_75::Float64 = quantile([x for x in y9 if x < 0], 1 - 0.75)

    return TMInput([
        # Raw pixels
        [x > 0 for x in raw];
        [x > raw_hist_25 for x in raw];
        [x > raw_hist_50 for x in raw];
        [x > raw_hist_75 for x in raw];

        # 3x3 convolution results
        [x > 0 for x in x3];
        [x > x3_hist_pos_25 for x in x3];
        [x > x3_hist_pos_34 for x in x3];
        [x > x3_hist_pos_50 for x in x3];
        [x > x3_hist_pos_75 for x in x3];
        [x < x3_hist_neg_25 for x in x3];
        [x < x3_hist_neg_34 for x in x3];
        [x < x3_hist_neg_50 for x in x3];
        [x < x3_hist_neg_75 for x in x3];

        [x > 0 for x in y3];
        [x > y3_hist_pos_25 for x in y3];
        [x > y3_hist_pos_34 for x in y3];
        [x > y3_hist_pos_50 for x in y3];
        [x > y3_hist_pos_75 for x in y3];
        [x < y3_hist_neg_25 for x in y3];
        [x < y3_hist_neg_34 for x in y3];
        [x < y3_hist_neg_50 for x in y3];
        [x < y3_hist_neg_75 for x in y3];

        # 5x5 convolution results
        [x > 0 for x in x5];
        [x > x5_hist_pos_25 for x in x5];
        [x > x5_hist_pos_34 for x in x5];
        [x > x5_hist_pos_50 for x in x5];
        [x > x5_hist_pos_75 for x in x5];
        [x < x5_hist_neg_25 for x in x5];
        [x < x5_hist_neg_34 for x in x5];
        [x < x5_hist_neg_50 for x in x5];
        [x < x5_hist_neg_75 for x in x5];

        [x > 0 for x in y5];
        [x > y5_hist_pos_25 for x in y5];
        [x > y5_hist_pos_34 for x in y5];
        [x > y5_hist_pos_50 for x in y5];
        [x > y5_hist_pos_75 for x in y5];
        [x < y5_hist_neg_25 for x in y5];
        [x < y5_hist_neg_34 for x in y5];
        [x < y5_hist_neg_50 for x in y5];
        [x < y5_hist_neg_75 for x in y5];

        # 7x7 convolution results
        [x > 0 for x in x7];
        [x > x7_hist_pos_25 for x in x7];
        [x > x7_hist_pos_34 for x in x7];
        [x > x7_hist_pos_50 for x in x7];
        [x > x7_hist_pos_75 for x in x7];
        [x < x7_hist_neg_25 for x in x7];
        [x < x7_hist_neg_34 for x in x7];
        [x < x7_hist_neg_50 for x in x7];
        [x < x7_hist_neg_75 for x in x7];

        [x > 0 for x in y7];
        [x > y7_hist_pos_25 for x in y7];
        [x > y7_hist_pos_34 for x in y7];
        [x > y7_hist_pos_50 for x in y7];
        [x > y7_hist_pos_75 for x in y7];
        [x < y7_hist_neg_25 for x in y7];
        [x < y7_hist_neg_34 for x in y7];
        [x < y7_hist_neg_50 for x in y7];
        [x < y7_hist_neg_75 for x in y7];

        # 9x9 convolution results
        [x > 0 for x in x9];
        [x > x9_hist_pos_25 for x in x9];
        [x > x9_hist_pos_34 for x in x9];
        [x > x9_hist_pos_50 for x in x9];
        [x > x9_hist_pos_75 for x in x9];
        [x < x9_hist_neg_25 for x in x9];
        [x < x9_hist_neg_34 for x in x9];
        [x < x9_hist_neg_50 for x in x9];
        [x < x9_hist_neg_75 for x in x9];

        [x > 0 for x in y9];
        [x > y9_hist_pos_25 for x in y9];
        [x > y9_hist_pos_34 for x in y9];
        [x > y9_hist_pos_50 for x in y9];
        [x > y9_hist_pos_75 for x in y9];
        [x < y9_hist_neg_25 for x in y9];
        [x < y9_hist_neg_34 for x in y9];
        [x < y9_hist_neg_50 for x in y9];
        [x < y9_hist_neg_75 for x in y9];
    ])
end

X_train::Vector{TMInput} = Vector{TMInput}(undef, length(x_train))
@threads for i in eachindex(x_train)
    X_train[i] = bools(x_train[i], x_train_conv_orient_x3[i], x_train_conv_orient_y3[i], x_train_conv_orient_x5[i], x_train_conv_orient_y5[i], x_train_conv_orient_x7[i], x_train_conv_orient_y7[i], x_train_conv_orient_x9[i], x_train_conv_orient_y9[i])
end

X_test::Vector{TMInput} = Vector{TMInput}(undef, length(x_test))
@threads for i in eachindex(x_test)
    X_test[i] = bools(x_test[i], x_test_conv_orient_x3[i], x_test_conv_orient_y3[i], x_test_conv_orient_x5[i], x_test_conv_orient_y5[i], x_test_conv_orient_x7[i], x_test_conv_orient_y7[i], x_test_conv_orient_x9[i], x_test_conv_orient_y9[i])
end

y_train = Int8.(y_train)
y_test = Int8.(y_test)

Serialization.serialize("/tmp/train", (X_train, y_train))
Serialization.serialize("/tmp/test", (X_test, y_test))
# exit()
X_train, y_train = Serialization.deserialize("/tmp/train")
X_test, y_test = Serialization.deserialize("/tmp/test")

println("Done.")

# CLAUSES = 2  # Best accuracy: 92.20% after 900 epochs
# T = 70
# S = 700
# L = 1000
# LF = 1000

# CLAUSES = 20  # Best accuracy: 93.41% after 820 epochs
# T = 100
# S = 700
# L = 200
# LF = 200

# CLAUSES = 20
# T = 100
# S = 2000
# L = 200
# LF = 200

# CLAUSES = 2000  # Best accuracy: 94.63% after 65 epochs
# T = 400
# S = 700
# L = 30
# LF = 30

CLAUSES = 8000  # Best accuracy: 94.74% after 11 epochs, Normal 94.68% test acc after 50 epochs.
T = 760
S = 700
L = 30
LF = 30

# CLAUSES = 200
# T = 200
# S = 700
# L = 100
# LF = 100


EPOCHS = 1000

# Training the TM model
tm = TMClassifier{eltype(y_train)}(CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=230)
tms = train!(tm, X_train, y_train, X_test, y_test, EPOCHS, shuffle=true, verbose=1, best_tms_size=1)

save(tms, "/tmp/tms.tm")
# tms = load("/tmp/tms.tm")
# tm, _ = combine(tms, 2, x_test, y_test)
# save(tm, "/tmp/tm2.tm")
# tm = load("/tmp/tm2.tm")
# tm = tms[1][1]
# accuracy(predict(tm, x_test), y_test) |> println
