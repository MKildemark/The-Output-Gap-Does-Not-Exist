
function tc_mwg(y, h, nDraws, burnin, mwg_const, σʸ; acc_target=0.25, adapt_interval=50)

# ----------------------------------------------------------------------------------------------------------------------
# Baseline empirical two-gap model used in the paper's US application.
#
# Observables are ordered as:
#   1. GDP
#   2. Employment
#   3. Unemployment
#   4. CPI inflation
#   5. UoM 1Y inflation expectations
#   6. SPF 1Y inflation expectations
#
# The reserved common-state block is:
#   1:2 -> efficient cycle (c_t^TG, c_t^{TG*})
#   3:4 -> cost-push cycle (u_t^TG, u_t^{TG*})
#   5   -> inflation trend (mu_t^pi)
#
# Observable-specific idiosyncratic blocks are appended after the first five columns.
# The raw Z / Z_plus masks below only identify the free pieces. The full NK-style
# loading restrictions from the paper are imposed in set_par! when the parameters
# are written back into the first five columns.
# ----------------------------------------------------------------------------------------------------------------------

     n = size(y)[2];


     # -----------------------------------------------------------------------------------------------------------------
     # Observation equations
     # -----------------------------------------------------------------------------------------------------------------

     d   = zeros(n);  # no intercepts
     Z   = [ones(n) zeros(6) [zeros(3); ones(3)] zeros(6) [zeros(3); ones(3)]]; # placeholder for [c_t^TG, c_t^{TG*}, u_t^TG, u_t^{TG*}, mu_t^pi]
     Z1a = kron(Matrix(I, 3, 3), [1, 0, 1])';         # GDP, employment, unemployment: idiosyncratic cycle + companion + trend
     Z1b = kron(Matrix(I, 2, 2), [1, 0, 1])';         # UoM/SPF expectations: idiosyncratic cycle + companion + trend
     Z2  = kron(Matrix(I, 1, 1), [1, 0])';            # Inflation: idiosyncratic cycle only

     Z  = [Z ex_blkdiag(Z1a, Z2, Z1b)];
     R  = zeros(n, n);         # no observation noise      # irregular components

     # Indices for observation equations
     d_ind = d .!= 0;  # no intercepts to be estimated
     Z_ind = zeros(size(Z)) .!= 0;
     R_ind = R .!= 0;  # no observation noise to be estimated

     # Free pieces of the common loading matrix.
     # The exact mapping into [delta_e, delta_u, delta_E, gamma_E, kappa] is done
     # in set_par!, not here.
     Z_ind[[2,3], [1]] .= true;

     # Positive-sign restricted pieces used for the expectation and inflation rows.
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

     T = cat(dims=[1,2], T_c, [T_ct for i=1:4]..., [T_c for i=1:1]..., [T_ct for i=1:2]...); # [efficient cycle], [cost-push cycle + trend], and idiosyncratic blocks
     Q = cat(dims=[1,2], T_c, [T_ct for i=1:4]..., [T_c for i=1:1]..., [T_ct for i=1:2]...); 

     # Indices for transition equations
     c_ind = c .!= 0;  # estimate drifts of gdp and employment trends
     T_ind = zeros(size(T)) .== 1;  # all T_ind is zero. No coefficients to be estimated. They are set in λ_ind and ρ_ind below
     
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
          mwg_main(par, h, nDraws, burnin, mwg_const, par_ind, σʸ; acc_target=acc_target, adapt_interval=adapt_interval);

     return distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_ind, par_size, distr_par;
end


