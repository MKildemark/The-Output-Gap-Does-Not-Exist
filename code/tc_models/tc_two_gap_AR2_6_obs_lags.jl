
function tc_mwg(y, h, nDraws, burnin, mwg_const, σʸ)

# ----------------------------------------------------------------------------------------------------------------------
# Alternative 6-observable two-gap specification with extra lagged common loadings.
#
# Observables are ordered as:
#   1. GDP
#   2. Employment
#   3. Unemployment
#   4. CPI inflation
#   5. UoM 1Y inflation expectations
#   6. SPF 1Y inflation expectations
#
# The state ordering matches the baseline empirical two-gap model:
#   1:2 -> efficient cycle (c_t^TG, c_t^{TG*})
#   3:4 -> cost-push cycle (u_t^TG, u_t^{TG*})
#   5   -> inflation trend (mu_t^pi)
#
# Relative to tc_two_gap_AR2_6_obs.jl, this variant allows additional lagged
# common-state terms in the measurement block. It is an exploratory extension,
# not the baseline specification described in the paper text.
# ----------------------------------------------------------------------------------------------------------------------

     n = size(y)[2];


     # -----------------------------------------------------------------------------------------------------------------
     # Observation equations
     # -----------------------------------------------------------------------------------------------------------------

     d   = zeros(n);  # no intercepts
     Z   = [ones(n) [0; ones(5)] [zeros(3); ones(3)] [zeros(3); ones(3)] [zeros(3); ones(3)]]; # placeholder common block with extra lag terms
     Z1a = kron(Matrix(I, 3, 3), [1, 0, 1])';         # GDP, employment, unemployment: idiosyncratic cycle + companion + trend
     Z1b = kron(Matrix(I, 2, 2), [1, 0, 1])';         # UoM/SPF expectations: idiosyncratic cycle + companion + trend
     Z2  = kron(Matrix(I, 1, 1), [1, 0])';            # Inflation: idiosyncratic cycle only

     Z  = [Z ex_blkdiag(Z1a, Z2, Z1b)];
     R  = zeros(n, n);         # no observation noise      # irregular components

     # Indices for observation equations
     d_ind = d .!= 0;  # no intercepts to be estimated
     Z_ind = zeros(size(Z)) .!= 0;
     R_ind = R .!= 0;  # no observation noise to be estimated

     # Free common-state loadings for the lagged-measurement variant.
     Z_ind[[2,3], [1,2]] .= true;
     Z_ind[[4,5],[2,4]] .= true;
   
     Z_plus_ind = zeros(size(Z)) .!= 0;
     Z_plus_ind[[4,5],[1]] .= true;
     Z_plus_ind[5,3] = true;
     Z_minus_ind = zeros(size(Z)) .!= 0;


     # -----------------------------------------------------------------------------------------------------------------
     # Transition equations
     # -----------------------------------------------------------------------------------------------------------------

     c              = zeros(size(Z)[2]);  #  constants in the transition equations (size = no. states)
     ind_trends     = [8; 11]; # GDP and employment trends have drifts
     c[ind_trends] .= 1;  # random-walk drifts for GDP and employment trends

     T_c     = convert(Array{Float64, 2}, [1 0; 0 0]);  # 2*2 transition block for cycle C and C+
     T_ct    = convert(Array{Float64, 2}, [1 0 0; 0 0 0; 0 0 1]); # 2*2 transition block for cycle C and C+ plus trend
     Q_c_ext = convert(Array{Float64, 2}, [1 0 0; 0 0 0; 0 0 0]);

     T = cat(dims=[1,2], T_c, [T_ct for i=1:4]..., [T_c for i=1:1]..., [T_ct for i=1:2]...); # same state transition structure as the baseline two-gap model
     Q = cat(dims=[1,2], T_c, [T_ct for i=1:4]..., [T_c for i=1:1]..., [T_ct for i=1:2]...); 


     H = Matrix{Float64}(I, size(T,1), size(T,1))


     # Indices for transition equations
     c_ind = c .!= 0;  # estimate drifts of gdp and employment trends
     T_ind = zeros(size(T)) .== 1;  # λ_ind and ρ_ind below

     Q_ind = Q .== 1; # estimate shock variances for cycles and trends
    
     Q_cov_ind = zeros(size(Q)) .!= 0;
     Q_cov_ind[1,3] = true; # estimate corr(xi^c_t, xi^u_t); set_par! converts it to covariance entries


     # Initial conditions for the non-stationary states
     P̄_c   = convert(Array{Float64, 2}, [0 0; 0 0]);
     P̄_ct  = convert(Array{Float64, 2}, [0 0 0; 0 0 0; 0 0 1]);
     P̄¹    = cat(dims=[1,2], [P̄_c for i=1:1]..., [P̄_ct for i=1:4]..., [P̄_c for i=1:1]..., [P̄_ct for i=1:2]...); # trends have diffuse initial conditions

     # Initial conditions
     α¹ = zeros(size(c));
     P¹       = zeros(size(P̄¹));

     # Trigonometric-cycle markers used by set_par! to rebuild each Harvey block
     λ_c   = convert(Array{Float64, 1}, [1; 0]);  
     λ_ct  = convert(Array{Float64, 1}, [1; 0; 0]);
     λ     = vcat([1; 0], [λ_ct for i=1:4]..., [λ_c for i=1:1]..., [λ_ct for i=1:2]...);
     ρ     = copy(λ);
     λ_ind = λ .!= 0;
     ρ_ind = copy(λ_ind);


     # -----------------------------------------------------------------------------------------------------------------
     # Metropolis-Within-Gibbs
     # -----------------------------------------------------------------------------------------------------------------

     par_ind = BoolParSsm(d_ind, Z_ind, Z_plus_ind, Z_minus_ind, R_ind, c_ind, T_ind, Q_ind, Q_cov_ind, λ_ind, ρ_ind);
     par     = ParSsm(permutedims(y), d, Z, R, c, T, Q, α¹, P¹, P̄¹, λ, ρ, 0.0, 0.0, 0.0);

     distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_size, distr_par =
          mwg_main(par, h, nDraws, burnin, mwg_const, par_ind, σʸ);

     return distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_ind, par_size, distr_par;
end
