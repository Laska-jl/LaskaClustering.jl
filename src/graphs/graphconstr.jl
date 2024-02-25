
# Building graphs from experiments
using SimpleWeightedGraphs: Graphs


"""
    connectedgraph(exp::AbstractExperiment, vars::Vector, σ = ℯ)

Construct a fully connected graph representing the `Clusters` of `exp`. Weights are calculated using the [`gaussiankernel`](@ref).
`vars` should be a `Vector` of `String`s, each matching a column name in the `info` DF of `exp`.
"""
function connectedgraph(exp::AbstractExperiment, vars::Vector, σ = ℯ)
    inf = deepcopy(info(exp, vcat(["cluster_id"], vars)))
    for c = 2:length(vars)+1
        inf[:, c] = LaskaCore.normalize(inf[:, c], extrema(inf[:, c]))
    end
    nclusters = size(inf, 1)
    vs = collect(transpose(Array(inf[:, 2:end])))
    adj = Matrix{Float64}(undef, nclusters, nclusters)
    adjmatrix!(adj, vs, σ)
    return SimpleWeightedGraph(adj)
end

"""
    adjmatrix!(A::AbstractArray, vars::AbstractArray, σ)

Populate the matrix `A` with the weights based off `vars` which should be a `Matrix` in which each column corresponds to a `Cluster`. `σ` controls the width of the [`gaussiankernel`](@ref).
"""
function adjmatrix!(A::AbstractArray, vars::AbstractArray, σ)
    for (i, j) in Iterators.product(1:size(vars, 2), 1:size(vars, 2))
        A[i, j] = gaussiankernel(vars[:, i], vars[:, j], σ)
    end
end


"""
    laprw(g::SimpleWeightedGraph)

Calculate the normalized Laplacian matrix ``L_{rw}``.

``L_{rw} = D^{-1}L = I - D^{-1}W``
"""
function laprw(g::SimpleWeightedGraph)
    D = collect(degree_matrix(g))
    W = collect(Graphs.weights(g))
    I - inv(D) * W
end
