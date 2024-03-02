
# Building graphs from experiments
using SimpleWeightedGraphs: Graphs


"""
    connectedgraph(exp::AbstractExperiment, vars::Vector, σ = ℯ)

Construct a fully connected graph representing the `Clusters` of `exp`. Weights are calculated using a [`gaussiankernel`](@ref).
`vars` should be a `Vector` of `String`s, each matching a column name in the `info` DF of `exp`. σ controls the width of the gaussian kernel.
"""
function connectedgraph(exp::AbstractExperiment, vars::Vector, σ=1)
    inf = deepcopy(info(exp, vars))
    for c = 1:length(vars)
        inf[:, c] = LaskaStats.rangenormalize(inf[:, c], extrema(inf[:, c]))
    end
    nclusters = size(inf, 1)
    vs = collect(transpose(Array(inf)))
    adj = Matrix{Float64}(undef, nclusters, nclusters)
    adjmatrix!(adj, vs, σ)
    return SimpleWeightedGraph(adj)
end

function knngraph(exp::AbstractExperiment, vars::Vector, k::Int, σ=1)
    inf = Array(info(exp, vars))
    for c = 1:length(vars)
        LaskaStats.rangenormalize!(inf[:, c], extrema(inf[:, c]))
    end
    inf = collect(transpose(inf))
    tree = KDTree(inf, leafsize=10)

end

function knnifyiadj!(adj::AbstractMatrix, k::Int, mutual::Bool)

end

"""
    adjmatrix!(A::AbstractArray, vars::AbstractArray, σ)

Populate `A` with weights based off `vars` which should be a `Matrix` in which each column corresponds to a `Cluster`. Weights are calculated using a gaussian kernel. `σ` controls the width of the [`gaussiankernel`](@ref).
"""
function adjmatrix!(A::Matrix, vars::Matrix, σ)
    for (i, j) in Iterators.product(1:size(vars, 2), 1:size(vars, 2))
        A[i, j] = gaussiankernel(vars[:, i], vars[:, j], σ)
    end
end

"""
    laprw(g::SimpleWeightedGraph)

Calculate the normalized Laplacian matrix ``L_{rw}``.

```math
L_{rw} = D^{-1}L = I - D^{-1}W
```
"""
function laprw(g::SimpleWeightedGraph)
    D = collect(degree_matrix(g))
    W = collect(Graphs.weights(g))
    I - inv(D) * W
end
