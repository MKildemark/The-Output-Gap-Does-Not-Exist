#=
This file is part of the replication code for: Hasenzagl, T., Pellegrino, F., Reichlin, L., & Ricco, G. (2020). A Model of the Fed's View on Inflation.
Please cite the paper if you are using any part of the code for academic work (including, but not limited to, conference and peer-reviewed papers).
=#

function set_par!(θ_bound, θ_unb, par, opt_transf, MIN, MAX, par_ind, par_size, prior_opt, apriori_rejection, σʸ)

# ----------------------------------------------------------------------------------------------------------------------
# Update the parameters of the state-space model
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
               if size(par.Z)[1] == 5 # full model 
                    # undo previous scale
                    par.Z[1:5, 1:4] .= Diagonal(vec(σʸ)) * par.Z[1:5, 1:4];
                    # fill parameters
                    par.Z[5,1:4] = copy(par.Z[4,1:4]); # common loading on obs 5 same as obs 4
                    # scale with 1/σʸ
                    par.Z[1:5, 1:4] .= Diagonal(1.0 ./ vec(σʸ)) * par.Z[1:5, 1:4]  # scale the common loadings with 1/σʸ to get common states in true scale
               elseif size(par.Z)[1] == 7 # full_inf model specific restriction
                    # undo previous scale
                    par.Z[1:7, 1:5] .= Diagonal(vec(σʸ)) * par.Z[1:7, 1:5];
                    # fill parameters
                    par.Z[7,1:4] = par.Z[6,1:4]; # common loading on obs 6 same as obs 7
                    par.Z[4,1:4] = par.Z[4,1:4].+par.Z[6,1:4];
                    par.Z[5,1:2] = par.Z[4,1:2];  
                    par.Z[5,1:5] = par.Z[5,1:5].*0; # no loading on common states for core inflation (temp)
                    # scale with 1/σʸ
                    par.Z[1:7, 1:5] .= Diagonal(1.0 ./ vec(σʸ)) * par.Z[1:7, 1:5]  # scale the common loadings with 1/σʸ to get common states in true scale
               elseif par_size.Z==2 # flex gap model specific restriction
                    par.Z[3,1:2] = par.Z[2,1:2]; # kappa on both eff and flex gap
               end
               par.logprior              = par.logprior + sum(logpdf.(prior_opt.N, par.Z[par_ind.Z .== true]));
               iend                      = iend+par_size.Z;
          end

          if par_size.Z_plus > 0 # par.Z below is correct
               par.Z[par_ind.Z_plus .== true] = θ_bound[iend+1:iend+par_size.Z_plus];
               par.logprior                        = par.logprior + sum(logpdf.(prior_opt.N_plus, par.Z[par_ind.Z_plus .== true]));
               iend                                = iend+par_size.Z_plus;
           end

          if par_size.Z_minus > 0 # par.Z below is correct
              par.Z[par_ind.Z_minus .== true] = θ_bound[iend+1:iend+par_size.Z_minus];
              par.logprior                          = par.logprior + sum(logpdf.(prior_opt.N_minus, par.Z[par_ind.Z_minus .== true]));
              iend                                  = iend+par_size.Z_minus;
          end
          
          # Transition equations

          if par_size.Q > 0
               par.Q[par_ind.Q .== true] = θ_bound[iend+1:iend+par_size.Q];
               par.logprior              = par.logprior + sum(logpdf.(prior_opt.IG, par.Q[par_ind.Q .== true]));
               iend                      = iend+par_size.Q;
          end

          if par_size.Q_cov > 0
               par.Q[par_ind.Q_cov .== true] = θ_bound[iend+1:iend+par_size.Q_cov];
               par.logprior                     = par.logprior + prior_opt.corr;
               iend                             = iend+par_size.Q_cov;
               # Set the correlation coefs in Q
               inds = findall(par_ind.Q_cov .== true);
               for I in inds
                    row, col = Tuple(I)              

                    ρ = par.Q[row, col]              # sampled correlation from θ

                    σi2 = par.Q[row, row]            # variance of the diagonals
                    σj2 = par.Q[col, col]

                    cov = ρ * sqrt(σi2 * σj2)        # correlation -> covariance

                    # Fill the four off-diagonals
                    par.Q[row,     col    ] = cov    # 
                    par.Q[col,     row    ] = cov    # cos–cos symmetric
                    par.Q[row + 1, col + 1] = cov    # sin–sin
                    par.Q[col + 1, row + 1] = cov    # sin–sin symmetric
               end

      
          end


          if par_size.c > 0
               par.c[par_ind.c .== true] = θ_bound[iend+1:iend+par_size.c];
               par.logprior              = par.logprior + sum(logpdf.(prior_opt.N, par.c[par_ind.c .== true]));
               iend                      = iend+par_size.c;
          end

          if par_size.T > 0 || par_size.λ > 0 || par_size.ρ > 0

               # Set T
               par.T[par_ind.T .== true] = θ_bound[iend+1:iend+par_size.T];
               par.logprior              = par.logprior + sum(logpdf.(prior_opt.N, par.T[par_ind.T .== true]));
               iend                      = iend+par_size.T;

               # Trigonometric states: update T, λ and ρ. Adjust Q and P¹
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
