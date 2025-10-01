
function tc_mwg(y, h, nDraws, burnin, mwg_const, σʸ)

# ----------------------------------------------------------------------------------------------------------------------
# Define the basic structure of the state-space parameters
# ----------------------------------------------------------------------------------------------------------------------

     n = size(y)[2];


     # -----------------------------------------------------------------------------------------------------------------
     # Observation equations
     # -----------------------------------------------------------------------------------------------------------------

     d   = zeros(n);  # no intercepts
     Z   = [ones(n) [0; ones(n-1)] [zeros(n-4); ones(4)] [zeros(4); ones(n-4)] [zeros(3); 1 ./ σʸ[end-3:end]]]; # the common factors in z
     Z1a = kron(Matrix(I, 3, 3), [1, 0, 1])';         # Idiosyncratic parts of y, e, u   (cycle plus trend)           # idio C, idio C+, idio trend
     Z1b = kron(Matrix(I, 2, 2), [1, 0, 1])';         # Idiosyncratic parts of inflation expectations (cycle plus trend)     # idio C, idio C+, idio trend
     Z2  = kron(Matrix(I, 2, 2), [1, 0])';            # idiosyncratic parts of infaltion (only idiosyncratic cycle)          # idio C, idio C+

     Z  = [Z ex_blkdiag(Z1a, Z2, Z1b)];
     R  = zeros(n, n);         # no observation noise      # irregular components

     # Indeces for observation equations  (where are parameters to be estimated)
     d_ind = d .!= 0;  # no intercepts to be estimated
     Z_ind = zeros(size(Z)) .!= 0;
     R_ind = R .!= 0;  # no observation noise to be estimated

     # Projections
     Z_ind[[2,3,4,6], [1,2]] .= true; # All        -> PC cycle, t and t-1  # business cycle and first lag loads on all 8 observations but with unity on y (first observation)
     Z_ind[[5,6],[3,4]]     .= true; # Expect.    -> PC cycle, t-2   # busines cycle lag 2 loads on expectations only
     Z_ind[4, 4] = true; # Prices     -> EP cycle, t and t-1  # energy cycle and first lag loads on oil (4th), infaltion expectations (5-6th) and inflation (7-8th) but with unity on oil

     # Z_plus_ind and Z_minus_ind (not used here)
     Z_plus_ind = zeros(size(Z)) .!= 0;
     Z_minus_ind = zeros(size(Z)) .!= 0;


     # -----------------------------------------------------------------------------------------------------------------
     # Transition equations
     # -----------------------------------------------------------------------------------------------------------------

     c              = zeros(size(Z)[2]);  #  constants in the transition equations (size = no. states)
     ind_trends     = [8; 11]; # GDP trend and EMPL trend state indexes
     c[ind_trends] .= 1;  # random walk drift for GDP and EMPL trends

     T_c     = convert(Array{Float64, 2}, [1 0; 0 0]);  # 2*2 transition block for cycle C and C+
     T_ct    = convert(Array{Float64, 2}, [1 0 0; 0 0 0; 0 0 1]); # 2*2 transition block for cycle C and C+ plus trend
     Q_c_ext = convert(Array{Float64, 2}, [1 0 0; 0 0 0; 0 0 0]);

     T = cat(dims=[1,2], T_c, [T_ct for i=1:4]..., [T_c for i=1:2]..., [T_ct for i=1:2]...); # PC(cycle), EP(cycle+trend),y(cycle+trend), e(cycle+trend), u(cycle+trend), inf(cycle), inf^c(cycle), UoM(cycle+trend), SPF(cycle+trend)
     Q = cat(dims=[1,2], T_c, [T_ct for i=1:4]..., [T_c for i=1:2]..., [T_ct for i=1:2]...); 

     # Indeces for transition equations
     c_ind = c .!= 0;  # estimate drifts of gdp and employment trends
     T_ind = zeros(size(T)) .== 1;  # all T_ind is zero. No coefficients to be estimated. They are set in λ_ind and ρ_ind below
     Q_ind = Q .== 1; # estiamte variances of shocks to cycles and trends. C and C+ have same variance
     Q[1,3] = 1; # allow correlation between shocks to Ψe (state 1) and Ψπ (state 3)
     Q_cov_ind = zeros(size(Q)) .!= 0;
     Q_cov_ind[1,3] = true; # estimate the covariance between shocks to Ψe (state 1) and Ψπ (state 3)


     # Initial conditions for the non-stationary states
     P̄_c   = convert(Array{Float64, 2}, [0 0; 0 0]);
     P̄_ct  = convert(Array{Float64, 2}, [0 0 0; 0 0 0; 0 0 1]);
     P̄¹    = cat(dims=[1,2], [P̄_c for i=1:1]..., [P̄_ct for i=1:4]..., [P̄_c for i=1:2]..., [P̄_ct for i=1:2]...); # trends have diffuse initial conditions

     # Initial conditions
     α¹ = zeros(size(c));
     P¹       = zeros(size(P̄¹));

     # Trigonometric states (indicates where the 2*2 cycle blocks start in T and Q. Needed to fill T and Q in set_par)
     λ_c   = convert(Array{Float64, 1}, [1; 0]);  
     λ_ct  = convert(Array{Float64, 1}, [1; 0; 0]);
     λ     = vcat([1; 0], [λ_ct for i=1:4]..., [λ_c for i=1:2]..., [λ_ct for i=1:2]...);
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
