function apply_two_gap_measurement_loadings!(par, σʸ, β)
     par.Z[1:6, 1:5] .= Diagonal(vec(σʸ)) * par.Z[1:6, 1:5]
     par.Z[1,1]       = 1.0
     par.Z[4:6, 5]   .= 1.0
     par.Z[4,3]       = 1.0
     par.Z[6,1:4]     = copy(par.Z[5,1:4])
     par.Z[4,1:4]    .= par.Z[4,1:4] .+ β^4 * par.Z[5,1:4]
     par.Z[1:6, 1:5] .= Diagonal(1.0 ./ vec(σʸ)) * par.Z[1:6, 1:5]
end

function uses_two_gap_measurement_loadings(par, par_size)
     return size(par.Z, 1) == 6 && par_size.λ > 0 && par_size.Z_plus > 0
end


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

     β = 1.0 # discount factor

     # -------------------- Z (unrestricted loadings) --------------------
     if par_size.Z > 0
          par.Z[par_ind.Z .== true] = θ_bound[iend+1:iend+par_size.Z]

          par.logprior += sum(logpdf.(prior_opt.N, par.Z[par_ind.Z .== true]))
          iend         += par_size.Z
     end

     # -------------------- Z_plus (sign-restricted) --------------------
     if par_size.Z_plus > 0
          par.Z[par_ind.Z_plus .== true] = θ_bound[iend+1:iend+par_size.Z_plus]

          par.logprior += sum(logpdf.(prior_opt.N_plus, par.Z[par_ind.Z_plus .== true]))
          iend         += par_size.Z_plus
     end

     # -------------------- Z_minus (negative-sign restricted) --------------------
     if par_size.Z_minus > 0
         par.Z[par_ind.Z_minus .== true] = θ_bound[iend+1:iend+par_size.Z_minus]
          par.logprior += sum(logpdf.(prior_opt.N_minus, par.Z[par_ind.Z_minus .== true]))
          iend         += par_size.Z_minus
     end

     if uses_two_gap_measurement_loadings(par, par_size)
          apply_two_gap_measurement_loadings!(par, σʸ, β)
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

             ρ = par.Q[row, col]

             σi2 = par.Q[row, row]
             σj2 = par.Q[col, col]

             cov = ρ * sqrt(σi2 * σj2)

             if par_size.λ > 0
                 par.Q[row,     col    ] = cov
                 par.Q[col,     row    ] = cov
                 par.Q[row + 1, col + 1] = cov
                 par.Q[col + 1, row + 1] = cov
             else
                 par.Q[row, col] = cov
                 par.Q[col, row] = cov
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

          end
     end

     # ------------------------------------------------------------------------------------------------------------
     # Log-likelihood and posterior
     # ------------------------------------------------------------------------------------------------------------

     kalman_diffuse!(par, 1, 0, 0)     # compute loglikelihood with diffuse init
     par.logprior     += sum(get_logjacobian(θ_unb, MIN, MAX, opt_transf))
     par.logposterior  = par.loglik + par.logprior
 end
