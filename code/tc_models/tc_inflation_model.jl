#=
This file is part of the replication code for: Hasenzagl, T., Pellegrino, F., Reichlin, L., & Ricco, G. (2020). A Model of the Fed's View on Inflation.
Please cite the paper if you are using any part of the code for academic work (including, but not limited to, conference and peer-reviewed papers).
=#

function tc_mwg(y, h, nDraws, burnin, mwg_const, σʸ)

# ----------------------------------------------------------------------------------------------------------------------
# Define the basic structure of the state-space parameters
# ----------------------------------------------------------------------------------------------------------------------

     n = size(y)[2];


     # -----------------------------------------------------------------------------------------------------------------
     # Observation equations
     # -----------------------------------------------------------------------------------------------------------------
     # obs: y, π
     # states: Ψ, Ψ*, τ^y, τ^π
     d   = zeros(n); 
     Z   = [1.0 0.0 1.0 0.0;
            1.0 0.0 0.0 1.0]; # the common factors in z


     R  = zeros(n, n);         # no observation noise      # irregular components

     # Indeces for observation equations  (where are parameters to be estimated)
     d_ind = d .!= 0;  # no intercepts to be estimated
     Z_ind = zeros(size(Z)) .!= 0;
     R_ind = R .!= 0;  # no observation noise to be estimated

     # Projections
     Z_ind[2, 1] = true; # All        -> inflation loading on gap

     # Z_plus_ind and Z_minus_ind (not used here)
     Z_plus_ind = zeros(size(Z)) .!= 0;
     Z_minus_ind = zeros(size(Z)) .!= 0;


     # -----------------------------------------------------------------------------------------------------------------
     # Transition equations
     # -----------------------------------------------------------------------------------------------------------------
     kx = size(Z,2);
     c              = zeros(kx);  #  constants in the transition equations (size = no. states)
     ind_trends     = [3]; # GDP drift
     c[ind_trends] .= 1;  # random walk drift for GDP 

     



     T = [1.0 0.0 0.0 0.0; #cycle
          0.0 0.0 0.0 0.0;
           0.0 0.0 1.0 0.0; # gdp trend
           0.0 0.0 0.0 1.0]; # inflation trend
     Q = [1.0 0.0 0.0 0.0; #cycle
          0.0 0.0 0.0 0.0;
           0.0 0.0 1.0 0.0; # gdp trend
           0.0 0.0 0.0 1.0]; # inflation trend (just an intercept, no shocks)

     # Indeces for transition equations
     c_ind = c .!= 0;  # estimate drifts of gdp 
     T_ind = zeros(size(T)) .== 1;  # all T_ind is zero. No coefficients to be estimated. They are set in λ_ind and ρ_ind below
     Q_ind = Q .== 1; # estiamte variances of shocks to cycles and trends. C and C+ have same variance which is done in set_par

     # Indicate wich state allow for correlated innovations
     Q_cov_ind = zeros(size(Q)) .!= 0; #no covariances


     # Initial conditions for the non-stationary states
     P̄¹    = zeros(kx,kx); P̄¹[3,3]=1; P̄¹[4,4]=1

     # Initial conditions
     α¹ = zeros(size(c));
     P¹       = zeros(size(P̄¹));

     # Trigonometric states (indicates where the 2*2 cycle blocks start in T and Q. Needed to fill T and Q in set_par)
    
     λ     = [1.0; 0; 0; 0]
     ρ     = copy(λ);
     λ_ind = λ .!= 0;
     ρ_ind = copy(λ_ind);


     # -----------------------------------------------------------------------------------------------------------------
     # Metropolis-Within-Gibbs
     # -----------------------------------------------------------------------------------------------------------------

     par_ind = BoolParSsm(d_ind, Z_ind, Z_plus_ind, Z_minus_ind, R_ind, c_ind, T_ind, Q_ind, Q_cov_ind, λ_ind, ρ_ind);
     par     = ParSsm(permutedims(y), d, Z, R, c, T, Q, α¹, P¹, P̄¹, λ, ρ, 0.0, 0.0, 0.0);

     distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_size, distr_par =
          mwg_main(par, h, nDraws, burnin, mwg_const, par_ind);

     return distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_ind, par_size, distr_par;
end
