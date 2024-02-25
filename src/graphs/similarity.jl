#-----------------------------------------------------
#
# Functions for representing similarity of 2 Clusters
#
#-----------------------------------------------------


# Distance metrics

@doc raw"""
    euclideandistance(x::Vector{T}, y::Vector{T}) where {T}

Calculate the euclidean distance between `x` and `y`.

``d(x,y) = \sqrt{(x_1 - y_1)^2 + (x_2-y_2)^2 + ... + (x_n - y_n)^2}``
"""
function euclideandistance(x::Vector{T}, y::Vector{T}) where {T}
    if !isequal(length(x), length(y))
        throw(ArgumentError("Input vectors should be of the same length"))
    end
    out = zero(T)
    @inbounds @simd for i in eachindex(x)
        out += (x[i] - y[i])^2
    end
    return sqrt(out)
end

@doc raw"""
    euclideandistance2(x::Vector{T}, y::Vector{T}) where {T}

Calculate the squared euclidean distance between `x` and `y`.

``d(x,y) = ||x - y||^2``
"""
function euclideandistance2(x::Vector{T}, y::Vector{T}) where {T}
    if !isequal(length(x), length(y))
        throw(ArgumentError("Input vectors should be of the same length"))
    end
    out = zero(T)
    @inbounds @simd for i in eachindex(x)
        out += (x[i] - y[i])^2
    end
    return out
end

@doc raw"""
    manhattandistance(x::Vector{T}, y::Vector{T}) where {T}

Calculate the manhattan distance between `x` and `y`.

``d(x,y) = |x_1 - y_1| + |x_2 - y_2| + ... + |x_n - y_n|``
"""
function manhattandistance(x::Vector{T}, y::Vector{T}) where {T}
    if !isequal(length(x), length(y))
        throw(ArgumentError("Input vectors should be of the same length"))
    end
    out = zero(T)
    @inbounds @simd for i in eachindex(x)
        out += abs(x[i] - y[i])
    end
    return out
end

@doc raw"""
    minkowskidistance(x::Vector{T}, y::Vector{T}, p) where {T}

Calculate the Minkowski distance between `x` and `y`.

``d(x,y) = \sum_{i=1}^{n}{|(x_i - y_i|^p)}^{\frac{1}{p}}``
"""
function minkowskidistance(x::Vector{T}, y::Vector{T}, p) where {T}
    if !isequal(length(x), length(y))
        throw(ArgumentError("Input vectors should be of the same length"))
    end
    out = zero(T)
    for i in eachindex(x)
        out += abs(x[i] - y[i])^p
    end
    return out^(1 / p)
end


@doc raw"""
    gaussiankernel(x::Vector{T}, y::Vector{T}, σ) where {T}

Applies the gaussian kernel to the euclidean distance between `x` and `y`. σ controls the width of the kernel with higher values resulting in a steeper dropoff.

``K(x,y)=exp\left ( \frac{-||x-y||^2}{2σ^2} \right )``
"""
function gaussiankernel(x::Vector{T}, y::Vector{T}, σ) where {T}
    exp(-euclideandistance2(x, y) / (2σ^2))
end
