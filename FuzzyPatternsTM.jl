module FuzzyPatternsTM

export TMInput, TMClassifier, train!, predict, accuracy, save, load

using Dates
using Random
using Base.Threads
using Serialization
using Statistics: mean, median
using Printf: @printf, @sprintf


unzip(a) = (getfield.(a, x) for x in fieldnames(eltype(a)))


abstract type AbstractTATeam end
abstract type AbstractTMClassifier end


mutable struct TATeam <: AbstractTATeam
    const include_limit::UInt8
    const state_min::UInt8
    const state_max::UInt8
    positive_clauses::Matrix{UInt8}
    negative_clauses::Matrix{UInt8}
    positive_included_literals::Vector{Vector{UInt16}}
    negative_included_literals::Vector{Vector{UInt16}}
    const clause_size::Int64

    function TATeam(clause_size::Int64, clauses_num::Int64, include_limit::Int64, state_min::Int64, state_max::Int64)
        positive_clauses = fill(UInt8(include_limit - 1), clause_size, floor(Int, clauses_num / 2))
        negative_clauses = fill(UInt8(include_limit - 1), clause_size, floor(Int, clauses_num / 2))
        positive_included_literals = fill([], floor(Int, clauses_num / 2))
        negative_included_literals = fill([], floor(Int, clauses_num / 2))
        return new(include_limit, state_min, state_max, positive_clauses, negative_clauses, positive_included_literals, negative_included_literals, clause_size)
    end
end


mutable struct TMClassifier <: AbstractTMClassifier
    clauses_num::Int64
    T::Int64
    R::Float64
    L::Int64
    LF::Int64
    const include_limit::Int64
    const state_min::Int64
    const state_max::Int64
    const clauses::Dict{Any, TATeam}

    function TMClassifier(clauses_num::Int64, T::Int64, R::Float64; states_num::Int64=256, include_limit::Int64=128, L::Int64=16, LF::Int64=4)
        return new(clauses_num, T, R, L, LF, include_limit, typemin(UInt8), states_num - 1, Dict())
    end
end


mutable struct TATeamCompiled <: AbstractTATeam
    positive_included_literals::Vector{Vector{UInt16}}
    negative_included_literals::Vector{Vector{UInt16}}

    function TATeamCompiled(clauses_num::Int64)
        positive_included_literals = fill([], floor(Int, clauses_num / 2))
        negative_included_literals = fill([], floor(Int, clauses_num / 2))
        return new(positive_included_literals, negative_included_literals)
    end
end


struct TMClassifierCompiled <: AbstractTMClassifier
    clauses_num::Int64
    T::Int64
    R::Float64
    L::Int64
    LF::Int64
    clauses::Dict{Any, TATeamCompiled}

    function TMClassifierCompiled(clauses_num::Int64, T::Int64, R::Float64, L::Int64, LF::Int64)
        return new(clauses_num, T, R, L, LF, Dict())
    end
end


abstract type AbstractTMInput <: AbstractVector{Bool} end

struct TMInput <: AbstractTMInput
    x::Vector{Bool}

    function TMInput(x::Vector{Bool}; negate::Bool=true)
        return negate ? new([x; [!_x for _x in x]]) : new(x)
    end
end

Base.IndexStyle(::Type{<:TMInput}) = IndexLinear()
Base.size(x::TMInput)::Tuple{Int64} = size(x.x)
Base.getindex(x::TMInput, i::Int)::Bool = x.x[i]


function initialize!(tm::TMClassifier, X::Vector{TMInput}, Y::Vector)
    for cls in collect(Set(Y))
        tm.clauses[cls] = TATeam(length(first(X)), tm.clauses_num, tm.include_limit, tm.state_min, tm.state_max)
    end
end


function check_clause(x::TMInput, literals::Vector{UInt16}, LF::Int64)::Int64
    c::Int64 = 0 < length(literals) < LF ? length(literals) : LF
    @inbounds @simd for i in eachindex(literals)
        if c <= 0
            return 0
        elseif !x[literals[i]]
            c -= 1
        end
    end
    return c
end


function vote(ta::AbstractTATeam, x::TMInput, LF::Int)::Tuple{Int64, Int64}
    pos = sum(check_clause(x, ta.positive_included_literals[i], LF) for i in eachindex(ta.positive_included_literals))
    neg = sum(check_clause(x, ta.negative_included_literals[i], LF) for i in eachindex(ta.negative_included_literals))
    return pos, neg
end


function feedback!(tm::TMClassifier, ta::TATeam, x::TMInput, clauses1::Matrix{UInt8}, clauses2::Matrix{UInt8}, literals1::Vector{Vector{UInt16}}, literals2::Vector{Vector{UInt16}}, positive::Bool)
    v::Int64 = clamp(-(vote(ta, x, tm.LF)...), -tm.T, tm.T)
    update::Float64 = (positive ? (tm.T - v) : (tm.T + v)) / (tm.T * 2)

    # Feedback 1
    @inbounds for (j, c) in enumerate(eachcol(clauses1))
        if (rand() < update)
            if check_clause(x, literals1[j], tm.LF) > 0  # Small change: Added `> 0`
                if (length(literals1[j]) <= tm.L)
                    @inbounds for i = 1:ta.clause_size
                        if (x[i] == true) && (c[i] < ta.state_max)
                            c[i] += one(UInt8)
                        end
                    end
                end
                @inbounds for i = 1:ta.clause_size
                    # No random
                    if (x[i] == false) && (c[i] < ta.include_limit) && (c[i] > ta.state_min)
                        c[i] -= one(UInt8)
                    end
                end
            else
                @inbounds for i = 1:ta.clause_size
                    # Here is one random only.
                    if (rand() > tm.R) && (c[i] > ta.state_min)
                        c[i] -= one(UInt8)
                    end
                end
            end
            literals1[j] = [@inbounds i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
        end
    end
    # Feedback 2
    @inbounds for (j, c) in enumerate(eachcol(clauses2))
        if (rand() < update)
            if check_clause(x, literals2[j], tm.LF) > 0  # Small change: Added `> 0`
                @inbounds for i = 1:ta.clause_size
                    # No random.
                    if (x[i] == false) && (c[i] < ta.include_limit)
                        c[i] += one(UInt8)
                    end
                end
                literals2[j] = [@inbounds i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
            end
        end
    end
end


function predict(tm::AbstractTMClassifier, x::AbstractTMInput)::Any
    best_vote::Int64 = typemin(Int64)
    best_cls::Any = nothing
    @inbounds for (cls, ta) in tm.clauses
        v::Int64 = -(vote(ta, x, tm.LF)...)
        if v > best_vote
            best_vote = v
            best_cls = cls
        end
    end
    return best_cls
end


function predict(tm::AbstractTMClassifier, X::Vector{TMInput})::Vector
    predicted::Vector = Vector{eltype(first(keys(tm.clauses)))}(undef, length(X))  # Predefine vector for @threads access
    @threads for i in eachindex(X)
        predicted[i] = predict(tm, X[i])
    end
    return predicted
end


function accuracy(predicted::Vector, Y::Vector)::Float64
    @assert eltype(predicted) == eltype(Y)
    @assert length(predicted) == length(Y)
    return sum(@inbounds 1 for (p, y) in zip(predicted, Y) if p == y; init=0) / length(Y)
end


function train!(tm::TMClassifier, x::TMInput, y::Any; shuffle::Bool=true)
    if shuffle
        classes = Random.shuffle(collect(keys(tm.clauses)))
    else
        classes = keys(tm.clauses)
    end
    for cls in classes
        if cls != y
            feedback!(tm, tm.clauses[y], x, tm.clauses[y].positive_clauses, tm.clauses[y].negative_clauses, tm.clauses[y].positive_included_literals, tm.clauses[y].negative_included_literals, true)
            feedback!(tm, tm.clauses[cls], x, tm.clauses[cls].negative_clauses, tm.clauses[cls].positive_clauses, tm.clauses[cls].negative_included_literals, tm.clauses[cls].positive_included_literals, false)
        end
    end
end


function train!(tm::TMClassifier, X::Vector{TMInput}, Y::Vector; shuffle::Bool=true)
    # If not initialized yet
    if length(tm.clauses) == 0
        initialize!(tm, X, Y)
    end
    if shuffle
        X, Y = unzip(Random.shuffle(collect(zip(X, Y))))
    end
    @threads for i in eachindex(Y)
        train!(tm, X[i], Y[i], shuffle=shuffle)
    end
end


function train!(tm::TMClassifier, x_train::Vector, y_train::Vector, x_test::Vector, y_test::Vector, epochs::Int64; shuffle::Bool=true, verbose::Int=1, best_tms_size::Int64=16, best_tms_compile::Bool=true)::Tuple{TMClassifier, Vector{Tuple{Float64, AbstractTMClassifier}}}
    @assert best_tms_size in 1:2000
    if verbose > 0
        println("\nRunning in $(nthreads()) threads.")
        println("Accuracy over $(epochs) epochs (Clauses: $(tm.clauses_num), T: $(tm.T), R: $(tm.R), L: $(tm.L), LF: $(tm.LF), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit)):\n")
    end
    best_tm = (0.0, nothing)
    best_tms = Tuple{Float64, AbstractTMClassifier}[]
    all_time = @elapsed begin
        for i in 1:epochs
            training_time = @elapsed train!(tm, x_train, y_train, shuffle=shuffle)
            testing_time = @elapsed begin
                acc = accuracy(predict(tm, x_test), y_test)
            end
            if acc >= first(best_tm)
                best_tm = (acc, deepcopy(tm))
            end
            push!(best_tms, (acc, best_tms_compile ? compile(tm, verbose=verbose - 1) : deepcopy(tm)))
            sort!(best_tms, by=first, rev=true)
            best_tms = best_tms[1:clamp(length(best_tms), length(best_tms), best_tms_size)]
            if verbose > 0
                @printf("#%s  Accuracy: %.2f%%  Best: %.2f%%  Training: %.3fs  Testing: %.3fs\n", i, acc * 100, best_tm[1] * 100, training_time, testing_time)
            end
        end
    end
    if verbose > 0
        println("\nDone. $(epochs) epochs (Clauses: $(tm.clauses_num), T: $(tm.T), R: $(tm.R), L: $(tm.L), LF: $(tm.LF), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit)).")
        elapsed = Time(0) + Second(floor(Int, all_time))
        @printf("Time elapsed: %s. Best accuracy was: %.2f%%.\n\n", elapsed, best_tm[1] * 100)
    end
    return best_tm[2], best_tms
end


function compile(tm::TMClassifier; verbose::Int=0)::TMClassifierCompiled
    if verbose > 0
        print("Compiling model... ")
        pos = []
        neg = []
    end
    all_time = @elapsed begin
        tmc = TMClassifierCompiled(tm.clauses_num, tm.T, tm.R, tm.L, tm.LF)
        for (cls, ta) in tm.clauses
            tmc.clauses[cls] = TATeamCompiled(tm.clauses_num)
            for (j, c) in enumerate(eachcol(ta.positive_clauses))
                tmc.clauses[cls].positive_included_literals[j] = [i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
                if verbose > 0
                    append!(pos, length(tmc.clauses[cls].positive_included_literals[j]))
                end
            end
            for (j, c) in enumerate(eachcol(ta.negative_clauses))
                tmc.clauses[cls].negative_included_literals[j] = [i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
                if verbose > 0
                    append!(neg, length(tmc.clauses[cls].negative_included_literals[j]))
                end
            end
        end
    end
    if verbose > 0
        @printf("Done. Time elapsed: %.3fs\n", all_time)
        pos_sum = sum(pos)
        neg_sum = sum(neg)
        println("Literals:")
        @printf("  Positive: %s, Negative: %s, Total: %s, Per clause: %.2f\n", pos_sum, neg_sum, (pos_sum + neg_sum), (pos_sum + neg_sum) / (length(tm.clauses) * tm.clauses_num))
        @printf("  Positive min: %s, max: %s, mean: %s, median: %s\n", minimum(pos), maximum(pos), mean(pos), median(pos))
        @printf("  Negative min: %s, max: %s, mean: %s, median: %s\n\n", minimum(neg), maximum(neg), mean(neg), median(neg))
    end
    return tmc
end


function compile(tms::Vector{Tuple{Float64, AbstractTMClassifier}})::Vector{Tuple{Float64, TMClassifierCompiled}}
    return [(acc, compile(tm, verbose=0)) for (acc, tm) in tms]
end


function save(tm::Union{AbstractTMClassifier, Tuple{Float64, AbstractTMClassifier}, Vector{Tuple{Float64, AbstractTMClassifier}}}, filepath::AbstractString)
    if !endswith(filepath, ".tm")
        filepath = string(filepath, ".tm")
    end
    print("Saving model to $(filepath)... ")
    Serialization.serialize(filepath, tm)
    println("Done.\n")
end


function load(filepath::AbstractString)
    if !endswith(filepath, ".tm")
        filepath = string(filepath, ".tm")
    end
    print("Loading model from $(filepath)... ")
    println("Done.\n")
    return Serialization.deserialize(filepath)
end

end # module
