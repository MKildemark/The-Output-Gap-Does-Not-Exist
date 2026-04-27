function tc_mwg(y, h, iter_init_adapt::Int64, iter_init_store::Int64, iter_main_adapt::Int64, iter_main_store::Int64,
                mwg_const, σʸ; acc_target::Float64=0.25, adapt_interval::Int64=50, t::Int64=0, end_oos::Int64=0)

# ----------------------------------------------------------------------------------------------------------------------
# Define the basic structure of the state-space parameters
# ----------------------------------------------------------------------------------------------------------------------

     n = size(y)[2]


     # -----------------------------------------------------------------------------------------------------------------
     # Observation equations
     # -----------------------------------------------------------------------------------------------------------------

     d   = zeros(n)
     Z   = [ones(n) zeros(n) [zeros(1); ones(3)./σʸ[2:4]]]
     Z1a = kron(Matrix(I, 1, 1), [1, 0, 1])'
     Z1b = kron(Matrix(I, 2, 2), [1, 0, 1])'
     Z2  = kron(Matrix(I, 1, 1), [1, 0])'

     Z  = [Z ex_blkdiag(Z1a, Z2, Z1b)]
     R  = zeros(n, n)

     # Indices for observation equations
     d_ind = d .!= 0
     Z_ind = zeros(size(Z)) .!= 0
     R_ind = R .!= 0

     # Projections
     Z_ind[[2,3,4], [1]] .= true

     # Z_plus_ind and Z_minus_ind
     Z_plus_ind = zeros(size(Z)) .!= 0
     Z_minus_ind = zeros(size(Z)) .!= 0


     # -----------------------------------------------------------------------------------------------------------------
     # Transition equations
     # -----------------------------------------------------------------------------------------------------------------

     c              = zeros(size(Z)[2])
     ind_trends     = [6]
     c[ind_trends] .= 1

     T_c     = convert(Array{Float64, 2}, [1 0; 0 0])
     T_ct    = convert(Array{Float64, 2}, [1 0 0; 0 0 0; 0 0 1])

     T = cat(dims=[1,2], [T_ct for i=1:2]..., [T_c for i=1:1]..., [T_ct for i=1:2]...)
     Q = cat(dims=[1,2], [T_ct for i=1:2]..., [T_c for i=1:1]..., [T_ct for i=1:2]...)

     # Indices for transition equations
     c_ind = c .!= 0
     T_ind = zeros(size(T)) .== 1

     Q_ind = Q .== 1

     Q_cov_ind = zeros(size(Q)) .!= 0

     # Initial conditions for the non-stationary states
     P̄_c   = convert(Array{Float64, 2}, [0 0; 0 0])
     P̄_ct  = convert(Array{Float64, 2}, [0 0 0; 0 0 0; 0 0 1])
     P̄¹    = cat(dims=[1,2], [P̄_ct for i=1:2]..., [P̄_c for i=1:1]..., [P̄_ct for i=1:2]...)

     # Initial conditions
     α¹ = zeros(size(c))
     P¹ = zeros(size(P̄¹))

     # Trigonometric states
     λ_c   = convert(Array{Float64, 1}, [1; 0])
     λ_ct  = convert(Array{Float64, 1}, [1; 0; 0])
     λ     = vcat([λ_ct for i=1:2]..., [λ_c for i=1:1]..., [λ_ct for i=1:2]...)
     ρ     = copy(λ)
     λ_ind = λ .!= 0
     ρ_ind = copy(λ_ind)


     # -----------------------------------------------------------------------------------------------------------------
     # Metropolis-Within-Gibbs
     # -----------------------------------------------------------------------------------------------------------------

     par_ind = BoolParSsm(d_ind, Z_ind, Z_plus_ind, Z_minus_ind, R_ind, c_ind, T_ind, Q_ind, Q_cov_ind, λ_ind, ρ_ind)
     par     = ParSsm(permutedims(y), d, Z, R, c, T, Q, α¹, P¹, P̄¹, λ, ρ, 0.0, 0.0, 0.0)

     distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_size, distr_par =
          mwg_main(par, h, iter_init_adapt, iter_init_store, iter_main_adapt, iter_main_store,
                   mwg_const, par_ind, σʸ;
                   acc_target=acc_target, adapt_interval=adapt_interval, t=t, end_oos=end_oos)

     return distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_ind, par_size, distr_par
end

function tc_mwg(y, h, nDraws, burnin, mwg_const, σʸ; acc_target::Float64=0.25, adapt_interval::Int64=50, t::Int64=0, end_oos::Int64=0)
     return tc_mwg(y, h, burnin[1], nDraws[1] - burnin[1], burnin[2], nDraws[2] - burnin[2], mwg_const, σʸ;
                   acc_target=acc_target, adapt_interval=adapt_interval, t=t, end_oos=end_oos)
end
