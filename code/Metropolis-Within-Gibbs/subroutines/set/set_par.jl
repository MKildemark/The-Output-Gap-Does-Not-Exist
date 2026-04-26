#=
This file is part of the replication code for: Hasenzagl, T., Pellegrino, F., Reichlin, L., & Ricco, G. (2020). A Model of the Fed's View on Inflation.
Please cite the paper if you are using any part of the code for academic work (including, but not limited to, conference and peer-reviewed papers).
=#

function is_two_gap_baseline_model(par_ind, par_size, par)
     return size(par.Z, 1) == 6 &&
            par_size.λ > 0 &&
            par_size.Z == 2 &&
            par_size.Z_plus == 3 &&
            sum(par_ind.Z) == 2 &&
            sum(par_ind.Z_plus) == 3 &&
            par_ind.Z[2, 1] && par_ind.Z[3, 1] &&
            par_ind.Z_plus[4, 1] && par_ind.Z_plus[5, 1] && par_ind.Z_plus[5, 3]
end

function is_two_gap_lag_model(par_ind, par_size, par)
     return size(par.Z, 1) == 6 &&
            par_size.λ > 0 &&
            par_size.Z == 8 &&
            par_size.Z_plus == 3 &&
            sum(par_ind.Z) == 8 &&
            sum(par_ind.Z_plus) == 3 &&
            par_ind.Z[2, 1] && par_ind.Z[2, 2] &&
            par_ind.Z[3, 1] && par_ind.Z[3, 2] &&
            par_ind.Z[4, 2] && par_ind.Z[4, 4] &&
            par_ind.Z[5, 2] && par_ind.Z[5, 4] &&
            par_ind.Z_plus[4, 1] && par_ind.Z_plus[5, 1] && par_ind.Z_plus[5, 3]
end

function rewrite_two_gap_baseline_common_block!(par, σʸ)
     common = Diagonal(vec(σʸ)) * copy(par.Z[1:6, 1:5])

     δe = common[2, 1]
     δu = common[3, 1]
     κ  = common[4, 1]
     δE = common[5, 1]
     γE = common[5, 3]

     common .= 0.0
     common[1, 1] = 1.0
     common[2, 1] = δe
     common[3, 1] = δu
     common[4, 1] = κ + δE
     common[4, 3] = 1.0 + γE
     common[4, 5] = 1.0
     common[5, 1] = δE
     common[5, 3] = γE
     common[5, 5] = 1.0
     common[6, 1] = δE
     common[6, 3] = γE
     common[6, 5] = 1.0

     par.Z[1:6, 1:5] .= Diagonal(1.0 ./ vec(σʸ)) * common
end

function rewrite_two_gap_lag_common_block!(par, σʸ)
     common = Diagonal(vec(σʸ)) * copy(par.Z[1:6, 1:5])

     δe1 = common[2, 1]
     δe2 = common[2, 2]
     δu1 = common[3, 1]
     δu2 = common[3, 2]

     κ0   = common[4, 1]
     κlag = common[4, 2]
     υlag = common[4, 4]

     δE0   = common[5, 1]
     δElag = common[5, 2]
     γE0   = common[5, 3]
     γElag = common[5, 4]

     common .= 0.0
     common[1, 1] = 1.0

     common[2, 1] = δe1
     common[2, 2] = δe2
     common[3, 1] = δu1
     common[3, 2] = δu2

     common[4, 1] = κ0 + δE0
     common[4, 2] = κlag + δElag
     common[4, 3] = 1.0 + γE0
     common[4, 4] = υlag + γElag
     common[4, 5] = 1.0

     common[5, 1] = δE0
     common[5, 2] = δElag
     common[5, 3] = γE0
     common[5, 4] = γElag
     common[5, 5] = 1.0

     common[6, 1] = δE0
     common[6, 2] = δElag
     common[6, 3] = γE0
     common[6, 4] = γElag
     common[6, 5] = 1.0

     par.Z[1:6, 1:5] .= Diagonal(1.0 ./ vec(σʸ)) * common
end

function set_par!(θ_bound, θ_unb, par, opt_transf, MIN, MAX, par_ind, par_size, prior_opt, apriori_rejection, σʸ)

# ----------------------------------------------------------------------------------------------------------------------
# Update the state-space object from the current parameter draw.
#
# Generic behavior:
# - write the free entries selected by par_ind into d, Z, R, c, T, Q
# - apply priors and transformation Jacobians
# - rebuild Harvey-cycle transition blocks from λ and ρ
#
# Local repository extensions relative to the original Hasenzagl code:
# - Q_cov stores free correlation parameters for selected shock pairs
# - the local 6-observable empirical two-gap models reserve the first five
#   columns of Z for the common-state block and impose their NK-style loading
#   restrictions here after the free coefficients have been inserted
# ----------------------------------------------------------------------------------------------------------------------


     if sum(ismissing.(θ_bound)) > 0 || sum(isinf.(θ_bound)) > 0 || sum(isnan.(θ_bound)) > 0 || sum(.~isreal.(θ_bound)) > 0
          apriori_rejection[1] = 1;

     else

          # ------------------------------------------------------------------------------------------------------------
          # Set state-space parameters and par.logprior
          # ------------------------------------------------------------------------------------------------------------

          iend         = 0;
          par.logprior = 0;

          # Observation equations

          if par_size.R > 0
               par.R[par_ind.R .== true] = θ_bound[1:par_size.R];
               par.logprior              = par.logprior + sum(logpdf.(prior_opt.IG, par.R[par_ind.R .== true]));
               iend                      = par_size.R;
          end

          if par_size.d > 0
               par.d[par_ind.d .== true] = θ_bound[iend+1:iend+par_size.d];
               par.logprior              = par.logprior + sum(logpdf.(prior_opt.N, par.d[par_ind.d .== true]));
               iend                      = iend+par_size.d;
          end

          if par_size.Z > 0
               par.Z[par_ind.Z .== true] = θ_bound[iend+1:iend+par_size.Z];
               if is_two_gap_baseline_model(par_ind, par_size, par)
                    rewrite_two_gap_baseline_common_block!(par, σʸ)
               elseif is_two_gap_lag_model(par_ind, par_size, par)
                    rewrite_two_gap_lag_common_block!(par, σʸ)
               end
               par.logprior              = par.logprior + sum(logpdf.(prior_opt.N, par.Z[par_ind.Z .== true]));
               iend                      = iend+par_size.Z;
          end

          if par_size.Z_plus > 0
               par.Z[par_ind.Z_plus .== true] = θ_bound[iend+1:iend+par_size.Z_plus];
               if is_two_gap_baseline_model(par_ind, par_size, par)
                    rewrite_two_gap_baseline_common_block!(par, σʸ)
               elseif is_two_gap_lag_model(par_ind, par_size, par)
                    rewrite_two_gap_lag_common_block!(par, σʸ)
               end
               par.logprior = par.logprior + sum(logpdf.(prior_opt.N_plus, par.Z[par_ind.Z_plus .== true]));
               iend         = iend+par_size.Z_plus;
          end

          if par_size.Z_minus > 0
               par.Z[par_ind.Z_minus .== true] = θ_bound[iend+1:iend+par_size.Z_minus];
               par.logprior                    = par.logprior + sum(logpdf.(prior_opt.N_minus, par.Z[par_ind.Z_minus .== true]));
               iend                            = iend+par_size.Z_minus;
          end

          # Transition equations

          if par_size.Q > 0
               par.Q[par_ind.Q .== true] = θ_bound[iend+1:iend+par_size.Q];
               par.logprior              = par.logprior + sum(logpdf.(prior_opt.IG, par.Q[par_ind.Q .== true]));
               iend                      = iend+par_size.Q;
          end

          if par_size.Q_cov > 0
               par.Q[par_ind.Q_cov .== true] = θ_bound[iend+1:iend+par_size.Q_cov];
               par.logprior                  = par.logprior + prior_opt.corr;
               iend                          = iend+par_size.Q_cov;

               inds = findall(par_ind.Q_cov .== true);
               for I in inds
                    row, col = Tuple(I)

                    # The sampler stores a correlation coefficient. Convert it to
                    # covariance using the variances already placed on the Q
                    # diagonal. For Harvey-cycle blocks, mirror the same
                    # covariance to the companion-star shock.
                    ρ = par.Q[row, col]

                    σi2 = par.Q[row, row]
                    σj2 = par.Q[col, col]

                    cov = ρ * sqrt(σi2 * σj2)

                    if par_size.λ > 0
                         par.Q[row, col] = cov
                         par.Q[col, row] = cov
                         par.Q[row + 1, col + 1] = cov
                         par.Q[col + 1, row + 1] = cov
                    else
                         par.Q[row, col] = cov
                         par.Q[col, row] = cov
                    end
               end
          end

          if par_size.c > 0
               par.c[par_ind.c .== true] = θ_bound[iend+1:iend+par_size.c];
               par.logprior              = par.logprior + sum(logpdf.(prior_opt.N, par.c[par_ind.c .== true]));
               iend                      = iend+par_size.c;
          end

          if par_size.T > 0 || par_size.λ > 0 || par_size.ρ > 0

               # Set any directly estimated T entries first
               par.T[par_ind.T .== true] = θ_bound[iend+1:iend+par_size.T];
               par.logprior              = par.logprior + sum(logpdf.(prior_opt.N, par.T[par_ind.T .== true]));
               iend                      = iend+par_size.T;

               # Harvey-cycle states: rebuild each 2x2 damped-rotation block
               # from its frequency λ and damping ρ, then update the matching
               # stationary variance in Q and P¹.
               if par_size.λ > 0 || par_size.ρ > 0
                    par.λ[par_ind.λ .== true] = θ_bound[iend+1:iend+par_size.λ];
                    par.ρ[par_ind.ρ .== true] = θ_bound[iend+par_size.λ+1:end];
                    par.logprior              = par.logprior + prior_opt.λ + prior_opt.ρ;

                    bool_trig = ((par_ind.λ .== true) .+ (par_ind.ρ .== true)) .> 0;
                    find_trig = findall(bool_trig);

                    for i=1:sum(bool_trig)
                         j = find_trig[i];

                         par.T[j:j+1, j:j+1]  = par.ρ[j]*[[cos(par.λ[j]) sin(par.λ[j])]; [-sin(par.λ[j]) cos(par.λ[j])]];
                         par.Q[j+1, j+1]      = par.Q[j, j];
                         par.P¹[j:j+1, j:j+1] = (par.Q[j, j] ./ (1-par.ρ[j]^2))*Array{Float64}(Matrix(I,2,2));
                    end
               end
          end


          # ------------------------------------------------------------------------------------------------------------
          # par.loglikelihood and par.logposterior
          # ------------------------------------------------------------------------------------------------------------

          kalman_diffuse!(par, 1, 0, 0); # estimate loglikelihood
          par.logprior     = par.logprior + sum(get_logjacobian(θ_unb, MIN, MAX, opt_transf));
          par.logposterior = par.loglik + par.logprior;
     end
end
