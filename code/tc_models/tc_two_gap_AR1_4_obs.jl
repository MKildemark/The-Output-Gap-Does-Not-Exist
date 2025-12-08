
function tc_mwg(y, h, nDraws, burnin, mwg_const, σʸ)

# ----------------------------------------------------------------------------------------------------------------------
# Define the basic structure of the state-space parameters
# ----------------------------------------------------------------------------------------------------------------------

     n = size(y)[2];


     # -----------------------------------------------------------------------------------------------------------------
     # Observation equations
     # -----------------------------------------------------------------------------------------------------------------

     d   = zeros(n);  # no intercepts
     Z   = [ones(n) [zeros(2); ones(2)] [zeros(2); ones(2)]]; # the common factors in z
     Z1a = kron(Matrix(I, 2, 2), [1, 1])';         # Idiosyncratic parts of y, e, u   (cycle plus trend)           # idio C, , idio trend
     Z1b = kron(Matrix(I, 1, 1), [1, 1])';         # Idiosyncratic parts of inflation expectations (cycle plus trend)     # idio C, , idio trend
     Z2  = kron(Matrix(I, 1, 1), [1])';            # idiosyncratic parts of infaltion (only idiosyncratic cycle)          # idio C,

     Z  = [Z ex_blkdiag(Z1a, Z2, Z1b)];
     R  = zeros(n, n);         # no observation noise      # irregular components

     # Indeces for observation equations  (where are parameters to be estimated)
     d_ind = d .!= 0;  # no intercepts to be estimated
     Z_ind = zeros(size(Z)) .!= 0;
     R_ind = R .!= 0;  # no observation noise to be estimated

     # Projections
     Z_ind[[2], [1]] .= true; # okun       
   

     # Z_plus_ind and Z_minus_ind 
     Z_plus_ind = zeros(size(Z)) .!= 0;
     Z_plus_ind[[3,4],[1]] .= true; # All       
     Z_plus_ind[4,2] = true; # Expect.    -> PC cycle, t-2   # busines cycle lag 2 loads on expectations only
     Z_minus_ind = zeros(size(Z)) .!= 0;


     # -----------------------------------------------------------------------------------------------------------------
     # Transition equations
     # -----------------------------------------------------------------------------------------------------------------

     c              = zeros(size(Z)[2]);  #  constants in the transition equations (size = no. states)
     ind_trends     = [5; 7]; # GDP trend and EMPL trend state indexes
     c[ind_trends] .= 1;  # random walk drift for GDP and EMPL trends

     T_c  = fill(1.0, 1, 1)                 # 1×1 for AR(1) cycle
     T_ct    = convert(Array{Float64, 2}, [1 0; 0 1]); # 2*2 transition block for cycle C and C+ plus trend

     T = cat(dims=[1,2], T_c, [T_ct for i=1:3]..., [T_c for i=1:1]..., [T_ct for i=1:1]...);
     Q = cat(dims=[1,2], T_c, [T_ct for i=1:3]..., [T_c for i=1:1]..., [T_ct for i=1:1]...); 

     # Indeces for transition equations
     c_ind = c .!= 0;  # estimate drifts of gdp and employment trends
     T_ind = zeros(size(T)) .!= 0;
     # make T_ind true on the diagonal for idx 1,2,4,6,8,9. estiamte ar1 coefficients
     for i in [1,2,4,6,8,9]
          T_ind[i,i] = true;
     end
     
     Q_ind = Q .== 1; # estiamte variances of shocks to cycles and trends. C and C+ have same variance

     Q_cov_ind = zeros(size(Q)) .!= 0;
     Q_cov_ind[1,2] = true; # estimate the covariance between shocks to Ψe (state 1) and Ψπ (state 3)

     H = Matrix{Float64}(I, size(T,1), size(T,1))
     H_ind = zeros(size(H)) .!= 0;
   

     # Initial conditions for the non-stationary states
     P̄_c  = zeros(1, 1)                    
     P̄_ct  = convert(Array{Float64, 2}, [0 0; 0 1]);
     P̄¹    = cat(dims=[1,2], [P̄_c for i=1:1]..., [P̄_ct for i=1:3]..., [P̄_c for i=1:1]..., [P̄_ct for i=1:1]...); # trends have diffuse initial conditions

     # Initial conditions
     α¹ = zeros(size(c));
     P¹       = zeros(size(P̄¹));

     # Trigonometric states (indicates where the 2*2 cycle blocks start in T and Q. Needed to fill T and Q in set_par)
   
     λ     = zeros(size(T,1)); # no trig cycles
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


