# Rational short-term expectations model (§1.2.2): RE-implied loadings for π, UoM, SPF
# Common states (first 5): g^e_1, g^e_2, g^π_1, g^π_2, μ^π
function tc_mwg(y, h, nDraws, burnin, mwg_const, σʸ)

# ----------------------------------------------------------------------------------------------------------------------
# Define the basic structure of the state-space parameters
# ----------------------------------------------------------------------------------------------------------------------

     n = size(y)[2];


     # -----------------------------------------------------------------------------------------------------------------
     # Observation equations
     # -----------------------------------------------------------------------------------------------------------------

     d   = zeros(n);  # no intercepts
     # Common states: columns 1-5 are (g^e_1, g^e_2, g^π_1, g^π_2, μ^π)
     # Initial structure: y loads on g^e_1, e/u load on g^e, π/UoM/SPF load on g^π and μ^π
     # These will be overwritten by RE logic in set_par.jl for rows 4-6
     Z   = [ones(n) zeros(n) zeros(n) zeros(n) [zeros(3); ones(2)]]; # the common factors in z
     Z1a = kron(Matrix(I, 3, 3), [1, 0, 1])';         # Idiosyncratic parts of y, e, u   (cycle plus trend)           # idio C, idio C+, idio trend
     Z1b = kron(Matrix(I, 1, 1), [1, 0])';         # Idiosyncratic parts of inflation expectations (cycle no trend)     # idio C, idio C+
     Z2  = kron(Matrix(I, 1, 1), [1, 0])';            # idiosyncratic parts of infaltion (only idiosyncratic cycle)          # idio C, idio C+

     Z  = [Z ex_blkdiag(Z1a, Z2, Z1b)];
     R  = zeros(n, n);         # no observation noise      # irregular components
     # add observation n

     # Indeces for observation equations  (where are parameters to be estimated)
     d_ind = d .!= 0;  # no intercepts to be estimated
     Z_ind = zeros(size(Z)) .!= 0;
     R_ind = R .!= 0;  # no observation noise to be estimated

     # Projections: e and u load on efficient gap cycle (both components)
     Z_ind[[2,3], [1]] .= true; # e and u load on g^e_1
     
     # Z_plus_ind: κ only (single parameter, will be used to compute RE loadings)
     # Place it in a location that gets overwritten anyway (e.g., Z[1,1] which is set to 1.0)
     Z_plus_ind = zeros(size(Z)) .!= 0;
     Z_plus_ind[1,1] = true;  # κ parameter (will be overwritten to 1.0 for identification)
     Z_minus_ind = zeros(size(Z)) .!= 0;


     # -----------------------------------------------------------------------------------------------------------------
     # Transition equations
     # -----------------------------------------------------------------------------------------------------------------

     c              = zeros(size(Z)[2]);  #  constants in the transition equations (size = no. states)
     # State ordering: 1-2 (g^e), 3-4 (g^π), 5 (μ^π), 6-8 (y), 9-11 (e), 12-14 (u), 15-16 (π), 17-19 (UoM), 20-22 (SPF)
     ind_trends     = [8; 11]; # GDP trend (state 8) and EMPL trend (state 11) state indexes
     c[ind_trends] .= 1;  # random walk drift for GDP and EMPL trends

     T_c     = convert(Array{Float64, 2}, [1 0; 0 0]);  # 2*2 transition block for cycle C and C+
     T_ct    = convert(Array{Float64, 2}, [1 0 0; 0 0 0; 0 0 1]); # 3*3 transition block for cycle C and C+ plus trend
     T_trend = convert(Array{Float64, 2}, [1 0; 0 0]); # transition block for random walk trend (no drift for μ^π)

     # T structure: common cycles (2×2 each), trend (1×1), then idio blocks
     # States: 1-2 (g^e), 3-4 (g^π), 5 (μ^π), 6-8 (y), 9-11 (e), 12-14 (u), 15-16 (π), 17-19 (UoM), 20-22 (SPF)
     T = cat(dims=[1,2], T_c, T_ct, [T_ct for i=1:3]..., T_c, [T_c for i=1:1]...);
     Q = cat(dims=[1,2], T_c, T_ct, [T_ct for i=1:3]..., T_c, [T_c for i=1:1]...); 

     # Indeces for transition equations
     c_ind = c .!= 0;  # estimate drifts of gdp and employment trends
     T_ind = zeros(size(T)) .== 1;  # all T_ind is zero. No coefficients to be estimated. They are set in λ_ind and ρ_ind below
     # T_ind[1,3] = true   # χ on cost push to eff gap
     
     Q_ind = Q .== 1; # estiamte variances of shocks to cycles and trends. C and C+ have same variance

     Q_cov_ind = zeros(size(Q)) .!= 0;
     Q_cov_ind[1,3] = true; # estimate the covariance between shocks to Ψe (state 1) and Ψπ (state 3)

     H = Matrix{Float64}(I, size(T,1), size(T,1))
     H_ind = zeros(size(H)) .!= 0;
     # H_ind[1,3] = true; # cost push shock can have contemporaneous effect on efficient gap cycle

     # Initial conditions for the non-stationary states
     P̄_c   = convert(Array{Float64, 2}, [0 0; 0 0]);
     P̄_ct  = convert(Array{Float64, 2}, [0 0 0; 0 0 0; 0 0 1]);
     # P̄ structure matches T: 2×2 for cycles, 1×1 for μ^π, then idio blocks
     P̄¹    = cat(dims=[1,2], [P̄_c for i=1:1]..., [P̄_ct for i=1:4]..., [P̄_c for i=1:1]..., [P̄_c for i=1:1]...);

     # Initial conditions
     α¹ = zeros(size(c));
     P¹       = zeros(size(P̄¹));

     # Trigonometric states (indicates where the 2*2 cycle blocks start in T and Q. Needed to fill T and Q in set_par)
     # Common cycles: states 1-2 (g^e) and 3-4 (g^π)
     # Idio cycles: y (6-7), e (9-10), u (12-13), π (15-16), UoM (17-18), SPF (20-21)
     λ_c   = convert(Array{Float64, 1}, [1; 0]);  # 2-element: first is λ, second is 0
     λ_ct  = convert(Array{Float64, 1}, [1; 0; 0]);  # 3-element: first is λ, rest are 0
     λ_trend = convert(Array{Float64, 1}, [0]);  # trend has no λ
     # Structure: [g^e_λ, g^e_0, g^π_λ, g^π_0, μ^π_0, y_λ, y_0, y_0, e_λ, e_0, e_0, u_λ, u_0, u_0, π_λ, π_0, UoM_λ, UoM_0, UoM_0, SPF_λ, SPF_0, SPF_0]
     λ     = vcat(λ_c, λ_c, λ_trend, [λ_ct for i=1:3]..., λ_c, [λ_c for i=1:1]...);
     ρ     = copy(λ);
     λ_ind = λ .!= 0;
     ρ_ind = copy(λ_ind);


     # -----------------------------------------------------------------------------------------------------------------
     # Metropolis-Within-Gibbs
     # -----------------------------------------------------------------------------------------------------------------

     par_ind = BoolParSsm(d_ind, Z_ind, Z_plus_ind, Z_minus_ind, R_ind, c_ind, T_ind, Q_ind, Q_cov_ind, H_ind, λ_ind, ρ_ind);
     par     = ParSsm(permutedims(y), d, Z, R, c, T, Q, H, α¹, P¹, P̄¹, λ, ρ, 0.0, 0.0, 0.0);

     distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_size, distr_par =
          mwg_main(par, h, nDraws, burnin, mwg_const, par_ind, σʸ);

     return distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_ind, par_size, distr_par;
end


