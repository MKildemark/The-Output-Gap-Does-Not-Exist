# Rational short-term expectations model (§1.2.2): RE-implied loadings for π, UoM, SPF
# Common states (first 5): g^e_1, g^e_2, g^π_1, g^π_2, μ^π
# Modified version: No idiosyncratic cycles for π and Eπ, white noise in R instead
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
     # NOTE: Removed Z2 and Z1b - no idiosyncratic cycles for π and Eπ
     # Pad Z1a to 5×9 by adding zeros for π and Eπ (rows 4-5)
     Z1a_padded = [Z1a; zeros(2, size(Z1a, 2))];  # 5×9 matrix with zeros for π and Eπ

     Z  = [Z Z1a_padded];  # Only include Z1a for y, e, u, zeros for π and Eπ
     R  = zeros(n, n);         # observation noise for π and Eπ
     # Add white noise for observations 4 (π) and 5 (Eπ)
     R[4,4] = 1.0;  # initial value for π observation noise
     R[5,5] = 1.0;  # initial value for Eπ observation noise

     # Indeces for observation equations  (where are parameters to be estimated)
     d_ind = d .!= 0;  # no intercepts to be estimated
     Z_ind = zeros(size(Z)) .!= 0;
     R_ind = zeros(size(R)) .!= 0;
     R_ind[4,4] = true;  # estimate observation noise variance for π
     R_ind[5,5] = true;  # estimate observation noise variance for Eπ

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
     # State ordering: 1-2 (g^e), 3-4 (g^π), 5 (μ^π), 6-8 (y), 9-11 (e), 12-14 (u)
     # NOTE: Removed states 15-16 (π cycle) and 17-18 (Eπ cycle)
     ind_trends     = [8; 11]; # GDP trend (state 8) and EMPL trend (state 11) state indexes
     c[ind_trends] .= 1;  # random walk drift for GDP and EMPL trends

     T_c     = convert(Array{Float64, 2}, [1 0; 0 0]);  # 2*2 transition block for cycle C and C+
     T_ct    = convert(Array{Float64, 2}, [1 0 0; 0 0 0; 0 0 1]); # 3*3 transition block for cycle C and C+ plus trend
     T_trend = convert(Array{Float64, 2}, [1 0; 0 0]); # transition block for random walk trend (no drift for μ^π)

     # T structure: common cycles (2×2 each), trend (1×1), then idio blocks for y, e, u only
     # States: 1-2 (g^e), 3-4 (g^π), 5 (μ^π), 6-8 (y), 9-11 (e), 12-14 (u)
     T = cat(dims=[1,2], T_c, T_ct, [T_ct for i=1:3]...);
     Q = cat(dims=[1,2], T_c, T_ct, [T_ct for i=1:3]...); 

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
     # P̄ structure matches T: 2×2 for cycles, 1×1 for μ^π, then idio blocks for y, e, u only
     P̄¹    = cat(dims=[1,2], [P̄_c for i=1:1]..., [P̄_ct for i=1:4]...);

     # Initial conditions
     α¹ = zeros(size(c));
     P¹       = zeros(size(P̄¹));

     # Trigonometric states (indicates where the 2*2 cycle blocks start in T and Q. Needed to fill T and Q in set_par)
     # Common cycles: states 1-2 (g^e) and 3-4 (g^π)
     # Idio cycles: y (6-7), e (9-10), u (12-13)
     # NOTE: Removed π (15-16) and Eπ (17-18) cycles
     λ_c   = convert(Array{Float64, 1}, [1; 0]);  # 2-element: first is λ, second is 0
     λ_ct  = convert(Array{Float64, 1}, [1; 0; 0]);  # 3-element: first is λ, rest are 0
     λ_trend = convert(Array{Float64, 1}, [0]);  # trend has no λ
     # Structure: [g^e_λ, g^e_0, g^π_λ, g^π_0, μ^π_0, y_λ, y_0, y_0, e_λ, e_0, e_0, u_λ, u_0, u_0]
     λ     = vcat(λ_c, λ_ct, [λ_ct for i=1:3]...);
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



