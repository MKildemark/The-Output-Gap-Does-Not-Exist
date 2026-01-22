

function set_par!(θ_bound, θ_unb, par, opt_transf, MIN, MAX, par_ind, par_size, prior_opt, apriori_rejection, σʸ)

     # ------------------------------------------------------------------------------------------------------------------
     # Update the parameters of the state-space model
     # ------------------------------------------------------------------------------------------------------------------
 
     if sum(ismissing.(θ_bound)) > 0 || sum(isinf.(θ_bound)) > 0 ||
        sum(isnan.(θ_bound)) > 0    || sum(.~isreal.(θ_bound)) > 0
 
         apriori_rejection[1] = 1
         return
     end
 
     # ---------------------------------------------------------------------------------------------------------------
     # Set state-space parameters and par.logprior
     # ---------------------------------------------------------------------------------------------------------------
 
     iend         = 0
     par.logprior = 0.0
 
     # -------------------- Observation noise R --------------------
     if par_size.R > 0
         par.R[par_ind.R .== true] = θ_bound[1:par_size.R]
         par.logprior += sum(logpdf.(prior_opt.IG, par.R[par_ind.R .== true]))
         iend = par_size.R
     end
 
     # -------------------- Intercepts d --------------------
     if par_size.d > 0
         par.d[par_ind.d .== true] = θ_bound[iend+1:iend+par_size.d]
         par.logprior += sum(logpdf.(prior_opt.N, par.d[par_ind.d .== true]))
         iend += par_size.d
     end
 
     # # -------------------- Discount factor ϕ (if any) --------------------
     # if par_size.ϕ > 0
     #     par.ϕ[par_ind.ϕ .== true] = θ_bound[iend+1:iend+par_size.ϕ]
     #     par.logprior += prior_opt.ϕ
     #     iend += par_size.ϕ
     # end
     β = 0.99 #discount factor
 
     # -------------------- Z (unrestricted loadings) --------------------
     if par_size.Z > 0
         par.Z[par_ind.Z .== true] = θ_bound[iend+1:iend+par_size.Z]
 
         if size(par.Z)[1] == 6 && par_size.λ > 0    # two-gap with AR(2), 6 obs
             par.Z[1:6, 1:5] .= Diagonal(vec(σʸ)) * par.Z[1:6, 1:5]
             par.Z[1,1]       = 1.0                  # y loads 1 on μ^e
             par.Z[4:6, 5]   .= 1.0                  # π, UoM, SPF load 1 on μ^π
             par.Z[4,3]       = 1.0
             par.Z[6,1:4]     = copy(par.Z[5,1:4])   # common loading on both Eπ
             par.Z[4,1:4]    .= par.Z[4,1:4] .+ β^4 * par.Z[5,1:4]  # expectations in baseline are 4 quarters ahead
             par.Z[1:6, 1:5] .= Diagonal(1.0 ./ vec(σʸ)) * par.Z[1:6, 1:5]
 
         end
 
         par.logprior += sum(logpdf.(prior_opt.N, par.Z[par_ind.Z .== true]))
         iend         += par_size.Z
     end
 
     # -------------------- Z_plus (sign-restricted) --------------------
     # this block fills in the sign restricted parameters (κ and in non-rational spec δ and γ related to inflation)
     if par_size.Z_plus > 0
         par.Z[par_ind.Z_plus .== true] = θ_bound[iend+1:iend+par_size.Z_plus]
 
         if size(par.Z)[1] == 6 && par_size.λ > 0
             par.Z[1:6, 1:5] .= Diagonal(vec(σʸ)) * par.Z[1:6, 1:5]
             par.Z[1,1]       = 1.0
             par.Z[4:6, 5]   .= 1.0
             par.Z[4,3]       = 1.0
             par.Z[6,1:4]     = copy(par.Z[5,1:4])
             par.Z[4,1:4]    .= par.Z[4,1:4] .+ β^4 * par.Z[5,1:4]  # expectations in baseline are 4 quarters ahead
            par.Z[1:6, 1:5] .= Diagonal(1.0 ./ vec(σʸ)) * par.Z[1:6, 1:5]
         end
 
         par.logprior += sum(logpdf.(prior_opt.N_plus, par.Z[par_ind.Z_plus .== true]))
         iend         += par_size.Z_plus
     end
 
     # -------------------- Z_minus (negative-sign restricted) --------------------
     if par_size.Z_minus > 0
         par.Z[par_ind.Z_minus .== true] = θ_bound[iend+1:iend+par_size.Z_minus]
         par.logprior += sum(logpdf.(prior_opt.N_minus, par.Z[par_ind.Z_minus .== true]))
         iend         += par_size.Z_minus
     end
 
     # -------------------- Q (variances of state shocks) --------------------
     if par_size.Q > 0
         par.Q[par_ind.Q .== true] = θ_bound[iend+1:iend+par_size.Q]
         par.logprior += sum(logpdf.(prior_opt.IG, par.Q[par_ind.Q .== true]))
         iend         += par_size.Q
     end
 
     # -------------------- Q_cov (correlations between state shocks) --------------------
     if par_size.Q_cov > 0
         par.Q[par_ind.Q_cov .== true] = θ_bound[iend+1:iend+par_size.Q_cov]
         par.logprior += prior_opt.corr
         iend         += par_size.Q_cov
 
         inds = findall(par_ind.Q_cov .== true)
         for I in inds
             row, col = Tuple(I)
 
             ρ = par.Q[row, col]               # “correlation” parameter in θ
 
             σi2 = par.Q[row, row]
             σj2 = par.Q[col, col]
 
             cov = ρ * sqrt(σi2 * σj2)         # correlation -> covariance
 
             if par_size.λ > 0
                 # trig-cycle (AR(2)): same covariance for cos & sin components
                 par.Q[row,     col    ] = cov
                 par.Q[col,     row    ] = cov
                 par.Q[row + 1, col + 1] = cov
                 par.Q[col + 1, row + 1] = cov
             else
                 # AR(1)
                 par.Q[row, col] = cov
                 par.Q[col, row] = cov
             end
         end
     end
 
     # -------------------- H (contemporaneous state mixing) --------------------
     if par_size.H > 0
         par.H[par_ind.H .== true] = θ_bound[iend+1:iend+par_size.H]
         par.logprior += sum(logpdf.(prior_opt.N, par.H[par_ind.H .== true]))
         iend         += par_size.H
 
         inds = findall(par_ind.H .== true)
         if par_size.λ > 0
             for I in inds
                 row, col = Tuple(I)
                 par.H[row + 1, col + 1] = par.H[row, col]
             end
         end
     end
 
     # -------------------- c (drifts in state equations) --------------------
     if par_size.c > 0
         par.c[par_ind.c .== true] = θ_bound[iend+1:iend+par_size.c]
         par.logprior += sum(logpdf.(prior_opt.N, par.c[par_ind.c .== true]))
         iend         += par_size.c
     end
 
     # -------------------- T, λ, ρ (state dynamics, trig cycles) --------------------
     if par_size.T > 0 || par_size.λ > 0 || par_size.ρ > 0
 
         if par_size.T > 0
             par.T[par_ind.T .== true] = θ_bound[iend+1:iend+par_size.T]
             par.logprior             += prior_opt.T
             iend                     += par_size.T
         end
 
         # Trigonometric states: update T, λ, ρ and adjust Q and P¹
         if par_size.λ > 0 || par_size.ρ > 0
             par.λ[par_ind.λ .== true] = θ_bound[iend+1:iend+par_size.λ]
             par.ρ[par_ind.ρ .== true] = θ_bound[iend+par_size.λ+1:end]
             par.logprior             += prior_opt.λ + prior_opt.ρ
 
             bool_trig = ((par_ind.λ .== true) .+ (par_ind.ρ .== true)) .> 0
             find_trig = findall(bool_trig)
 
             for i = 1:sum(bool_trig)
                 j = find_trig[i]
 
                 par.T[j:j+1, j:j+1] =
                     par.ρ[j] * [ cos(par.λ[j])  sin(par.λ[j]);
                                 -sin(par.λ[j])  cos(par.λ[j]) ]
 
                 par.Q[j+1, j+1]      = par.Q[j, j]
                 par.P¹[j:j+1, j:j+1] = (par.Q[j, j] / (1 - par.ρ[j]^2)) * Matrix{Float64}(I, 2, 2)
             end
 
            ########################################################################
            #  >>> Rational NKPC BLOCK <<<
            ########################################################################

            # Rational expectations implementation for §1.2.2 (short-term) and §1.2.3 (long-term)
            if (size(par.Z, 1) == 6 || size(par.Z, 1) == 5) && par_size.λ > 0 && par_size.Z_plus == 1

                n_obs = size(par.Z, 1)

                # κ is the *first* Z_plus parameter in θ_bound.
                # Note: ϕ is not used (β is fixed at 0.99), so it's not in the parameter vector
                idx_κ = par_size.R + par_size.d + par_size.Z + 1
                κ     = θ_bound[idx_κ]

                # Harvey cycle blocks in T:
                # gap cycle (g^e): states 1–2
                # cost-push cycle (g^π): states 3–4
                j_gap  = 1
                j_cost = 3

                Φ_gap  = par.T[j_gap:j_gap+1,  j_gap:j_gap+1]
                Φ_cost = par.T[j_cost:j_cost+1, j_cost:j_cost+1]

                I2     = Matrix{Float64}(I, 2, 2)

                # RE NKPC loadings (current period):
                # α_e'  = κ e1' (I - β Φ_gap)^(-1)
                # α_π'  =     e1' (I - β Φ_cost)^(-1)
                A_gap  = κ * [1.0 0.0] * inv(I2 .- β * Φ_gap)    # 1×2 row vector
                A_cost =      [1.0 0.0] * inv(I2 .- β * Φ_cost)  # 1×2 row vector

                # Rescale Z to original units before overwriting
                par.Z[1:n_obs, 1:5] .= Diagonal(vec(σʸ)) * par.Z[1:n_obs, 1:5]

                # Identification constraints
                par.Z[1, 1] = 1.0  # y loads 1 on first efficient-gap component

                # Inflation row (y, e, u, π, ...) -> π is row 4
                row_π = 4
                
                # Overwrite π row loadings on the two Harvey cycles (current period)
                par.Z[row_π, j_gap:j_gap+1]   .= A_gap[1, 1:2]
                par.Z[row_π, j_cost:j_cost+1] .= A_cost[1, 1:2]
                par.Z[row_π, 5] = 1.0  # Enforce the shared inflation trend loading

                # Short-term expectations (h=4): UoM and SPF rows
                if n_obs == 6
                    row_uom = 5  # UoM expected inflation
                    row_spf = 6  # SPF expected inflation
                    H_ST    = 4  # 4 quarters ahead

                    # Compute h=4 horizon loadings: α_e'(4) = α_e' Φ_gap^4, α_π'(4) = α_π' Φ_cost^4
                    A_gap_ST  = A_gap  * (Φ_gap^H_ST)   # 1×2
                    A_cost_ST = A_cost * (Φ_cost^H_ST)   # 1×2

                    # Overwrite UoM and SPF rows with h=4 loadings
                    par.Z[row_uom, j_gap:j_gap+1]   .= A_gap_ST[1, 1:2]
                    par.Z[row_uom, j_cost:j_cost+1] .= A_cost_ST[1, 1:2]
                    par.Z[row_uom, 5]                = 1.0

                    par.Z[row_spf, j_gap:j_gap+1]   .= A_gap_ST[1, 1:2]
                    par.Z[row_spf, j_cost:j_cost+1] .= A_cost_ST[1, 1:2]
                    par.Z[row_spf, 5]                = 1.0
                end

                # Long-term expectations (h=40): if only 5 observables, last row is long-term
                if n_obs == 5
                    row_LT = 5
                    H_LT   = 120

                    A_gap_LT  = A_gap  * (Φ_gap^H_LT)
                    A_cost_LT = A_cost * (Φ_cost^H_LT)

                    par.Z[row_LT, j_gap:j_gap+1]   .= A_gap_LT[1, 1:2]
                    par.Z[row_LT, j_cost:j_cost+1] .= A_cost_LT[1, 1:2]
                    par.Z[row_LT, 5]                = 1.0
                end
               
                # Scale back to standardized units
                par.Z[1:n_obs, 1:5] .= Diagonal(1.0 ./ vec(σʸ)) * par.Z[1:n_obs, 1:5]
           
            end
         end
     end
 
     # ------------------------------------------------------------------------------------------------------------
     # Log-likelihood and posterior
     # ------------------------------------------------------------------------------------------------------------
 
     kalman_diffuse!(par, 1, 0, 0)     # compute loglikelihood with diffuse init
     par.logprior     += sum(get_logjacobian(θ_unb, MIN, MAX, opt_transf))
     par.logposterior  = par.loglik + par.logprior
 end
 