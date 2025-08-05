include("../../FuzzyPatternTM.jl")

using Serialization
using .FuzzyPatternTM: TMClassifier, train!


X_train, y_train = Serialization.deserialize("/tmp/train")
X_test, y_test = Serialization.deserialize("/tmp/test")

# CLAUSES = 20
# T = 100
# S = 700
# L = 200
# LF = 200

CLAUSES = 8000  # Best accuracy: 94.74% after 11 epochs, Normal 94.68% test acc after 50 epochs.
T = 700
S = 700
L = 30
LF = 30

EPOCHS = 1000

# Training the TM model
tm = TMClassifier{eltype(y_train)}(CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=230)
tms = train!(tm, X_train, y_train, X_test, y_test, EPOCHS, shuffle=true, verbose=1, best_tms_size=1)
