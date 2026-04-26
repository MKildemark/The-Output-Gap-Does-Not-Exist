function tc_mwg(y, h, nDraws, burnin, mwg_const, σʸ; acc_target=0.25, adapt_interval=50)

# ----------------------------------------------------------------------------------------------------------------------
# Local 6-observable one-gap benchmark with Okun-style real activity block.
#
# Observables are ordered as:
#   1. GDP
#   2. Employment
#   3. Unemployment
#   4. CPI inflation
#   5. UoM 1Y inflation expectations
#   6. SPF 1Y inflation expectations
#
# Common states:
#   1:2 -> one common gap cycle written in Harvey-cycle form
#   3   -> common inflation trend
#
# This is the one-gap comparator used against the empirical two-gap model.
# ----------------------------------------------------------------------------------------------------------------------

     n = size(y)[2];


     # -----------------------------------------------------------------------------------------------------------------
     # Observation equations
     # -----------------------------------------------------------------------------------------------------------------

     d   = zeros(n);  # no intercepts
     Z   = [ones(n) zeros(n) [zeros(3); ones(3)./σʸ[4:6]]]; # common gap cycle plus common inflation trend
     Z1a = kron(Matrix(I, 3, 3), [1, 0, 1])';         # GDP, employment, unemployment: idiosyncratic cycle + companion + trend
     Z1b = kron(Matrix(I, 2, 2), [1, 0, 1])';         # UoM/SPF expectations: idiosyncratic cycle + companion + trend
     Z2  = kron(Matrix(I, 1, 1), [1, 0])';            # Inflation: idiosyncratic cycle only

     Z  = [Z ex_blkdiag(Z1a, Z2, Z1b)];  # full Z matrix
     R  = zeros(n, n);         # no observation noise      # irregular components

     # Indices for observation equations
     d_ind = d .!= 0;  # no intercepts to be estimated
     Z_ind = zeros(size(Z)) .!= 0;
     R_ind = R .!= 0;  # no observation noise to be estimated

     # Estimated loadings on the common one-gap factor.
     # GDP is normalized to load with coefficient one.
     Z_ind[[2,3,4,5,6], [1]] .= true;

     # Z_plus_ind and Z_minus_ind (not used here)
     Z_plus_ind = zeros(size(Z)) .!= 0;
     Z_minus_ind = zeros(size(Z)) .!= 0;


     # -----------------------------------------------------------------------------------------------------------------
     # Transition equations
     # -----------------------------------------------------------------------------------------------------------------

     c              = zeros(size(Z)[2]);  #  constants in the transition equations (size = no. states)
     ind_trends     = [6,9]; # GDP and employment trends have drifts
     c[ind_trends] .= 1;  # random-walk drifts for GDP and employment trends

     T_c     = convert(Array{Float64, 2}, [1 0; 0 0]);  # 2*2 transition block for cycle C and C+
     T_ct    = convert(Array{Float64, 2}, [1 0 0; 0 0 0; 0 0 1]); # 2*2 transition block for cycle C and C+ plus trend
     Q_c_ext = convert(Array{Float64, 2}, [1 0 0; 0 0 0; 0 0 0]);

     T = cat(dims=[1,2], [T_ct for i=1:4]..., [T_c for i=1:1]..., [T_ct for i=1:2]...); 
     Q =  cat(dims=[1,2], [T_ct for i=1:4]..., [T_c for i=1:1]..., [T_ct for i=1:2]...); 

     # Indices for transition equations
     c_ind = c .!= 0;  # estimate drifts of gdp and employment trends
     T_ind = zeros(size(T)) .== 1;  # all T_ind is zero. No coefficients to be estimated. They are set in λ_ind and ρ_ind below

     Q_ind = Q .== 1; # estimate shock variances for cycles and trends

     Q_cov_ind = zeros(size(Q)) .!= 0;


     # Initial conditions for the non-stationary states
     P̄_c   = convert(Array{Float64, 2}, [0 0; 0 0]);
     P̄_ct  = convert(Array{Float64, 2}, [0 0 0; 0 0 0; 0 0 1]);
     P̄¹    = cat(dims=[1,2], [P̄_ct for i=1:4]..., [P̄_c for i=1:1]..., [P̄_ct for i=1:2]...); # trends have diffuse initial conditions

     # Initial conditions
     α¹ = zeros(size(c));
     P¹       = zeros(size(P̄¹));

     # Trigonometric-cycle markers used by set_par! to rebuild each Harvey block
     λ_c   = convert(Array{Float64, 1}, [1; 0]);  
     λ_ct  = convert(Array{Float64, 1}, [1; 0; 0]);
     λ     = vcat([λ_ct for i=1:4]..., [λ_c for i=1:1]..., [λ_ct for i=1:2]...);
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




