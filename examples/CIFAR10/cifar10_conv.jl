include("../../src/FuzzyPatternTM_32b.jl")


using Serialization
using .FuzzyPatternTM: TMClassifier, train!


X_train, y_train = Serialization.deserialize("/tmp/CIFAR10_train")
X_test, y_test = Serialization.deserialize("/tmp/CIFAR10_test")


CLAUSES = 20  # (69%+ acc)
T = 1600
S = 1000
L = 4000
LF = 4000

# CLAUSES = 200
# T = 1500
# S = 1000
# L = 1000
# LF = 1000

# CLAUSES = 200
# T = 2500
# S = 1000
# L = 1000
# LF = 1000

# CLAUSES = 2000
# T = 10000  # 2200
# S = 1000   # 1000
# L = 1000   # 200
# LF = 1000  # 200

# CLAUSES = 2000
# T = 2200
# S = 1000
# L = 200
# LF = 200

EPOCHS = 1000

# Training the TM model
tm = TMClassifier{eltype(y_train)}(CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=240)
tms = train!(tm, X_train, y_train, X_test, y_test, EPOCHS, best_tms_size=0, shuffle=true, batch=true, verbose=1)
