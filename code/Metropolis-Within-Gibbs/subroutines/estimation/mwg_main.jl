#=
This file is part of the replication code for: Hasenzagl, T., Pellegrino, F., Reichlin, L., & Ricco, G. (2020). A Model of the Fed's View on Inflation.
Please cite the paper if you are using any part of the code for academic work (including, but not limited to, conference and peer-reviewed papers).
=#

function mwg_main(par::ParSsm, h::Int64,
                  iter_init_adapt::Int64, iter_init_store::Int64, iter_main_adapt::Int64, iter_main_store::Int64,
                  mwg_const::Array{Float64, 1}, par_ind::BoolParSsm, σʸ;
                  acc_target::Float64=0.25, adapt_interval::Int64=50, t=0::Int64, end_oos=0::Int64)

# ----------------------------------------------------------------------------------------------------------------------
# Metropolis-Within-Gibbs algorithm: mainframe
# ----------------------------------------------------------------------------------------------------------------------

     for j in axes(par_ind.Z, 2), i in axes(par_ind.Z, 1)
          if (par_ind.Z[i,j] == true) && ((par_ind.Z[i,j] == par_ind.Z_plus[i,j]) || (par_ind.Z[i,j] == par_ind.Z_minus[i,j]))
               error("Each measurement equation coefficient can be either unrestricted or with a specific sign restriction - not both!")
          elseif (par_ind.Z_plus[i,j] == true) && (par_ind.Z_plus[i,j] == par_ind.Z_minus[i,j])
               error("Each measurement equation coefficient can be either restricted to be positive or negative - not both!")
          end
     end

     par_size = SizeParSsm(sum(par_ind.d),
                           sum(sum(par_ind.Z)),
                           sum(sum(par_ind.Z_plus)),
                           sum(sum(par_ind.Z_minus)),
                           sum(sum(par_ind.R)),
                           sum(par_ind.c),
                           sum(sum(par_ind.T)),
                           sum(sum(par_ind.Q)),
                           sum(sum(par_ind.Q_cov)),
                           sum(par_ind.λ),
                           sum(par_ind.ρ),
                           sum(par_ind.d) + sum(sum(par_ind.Z)) + sum(sum(par_ind.Z_plus)) +
                           sum(sum(par_ind.Z_minus)) + sum(sum(par_ind.R)) +
                           sum(par_ind.c) + sum(sum(par_ind.T)) + sum(sum(par_ind.Q)) +
                           sum(sum(par_ind.Q_cov)) + sum(par_ind.λ) + sum(par_ind.ρ))

     xi              = 1e-3
     MIN_var         = 0
     MIN_coeff       = -Inf
     MIN_coeff_plus  = 0
     MIN_coeff_minus = -Inf
     MIN_λ           = xi
     MIN_ρ           = xi
     MIN_corr        = -0.99
     MAX_var         = Inf
     MAX_coeff       = Inf
     MAX_coeff_plus  = Inf
     MAX_coeff_minus = 0
     MAX_λ           = pi
     MAX_ρ           = 0.99
     MAX_corr        = 0.01

     MIN = [MIN_var*ones(par_size.R);
            MIN_coeff*ones(par_size.d + par_size.Z);
            MIN_coeff_plus*ones(par_size.Z_plus);
            MIN_coeff_minus*ones(par_size.Z_minus);
            MIN_var*ones(par_size.Q);
            MIN_corr*ones(par_size.Q_cov);
            MIN_coeff*ones(par_size.c);
            MIN_ρ*ones(par_size.T);
            MIN_λ*ones(par_size.λ);
            MIN_ρ*ones(par_size.ρ)]

     MAX = [MAX_var*ones(par_size.R);
            MAX_coeff*ones(par_size.d + par_size.Z);
            MAX_coeff_plus*ones(par_size.Z_plus);
            MAX_coeff_minus*ones(par_size.Z_minus);
            MAX_var*ones(par_size.Q);
            MAX_corr*ones(par_size.Q_cov);
            MAX_coeff*ones(par_size.c);
            MAX_ρ*ones(par_size.T);
            MAX_λ*ones(par_size.λ);
            MAX_ρ*ones(par_size.ρ)]

     prior_opt = PriorOpt(Normal(0, 1/xi),
                          Truncated(Normal(0, 1/xi), MIN_coeff_plus, MAX_coeff_plus),
                          Truncated(Normal(0, 1/xi), MIN_coeff_minus, MAX_coeff_minus),
                          InverseGamma(3, 1),
                          par_size.T*logpdf.(Uniform(MIN_ρ, MAX_ρ), MIN_ρ),
                          par_size.λ*logpdf.(Uniform(MIN_λ, MAX_λ), MIN_λ),
                          par_size.ρ*logpdf.(Uniform(MIN_ρ, MAX_ρ), MIN_ρ),
                          par_size.Q_cov*logpdf.(Uniform(MIN_corr, MAX_corr), MIN_corr))

     opt_transf = convert(Array{Int64, 1}, [1*ones(par_size.R);
                                            2*ones(par_size.d + par_size.Z);
                                            1*ones(par_size.Z_plus);
                                            0*ones(par_size.Z_minus);
                                            1*ones(par_size.Q);
                                            3*ones(par_size.Q_cov);
                                            2*ones(par_size.c);
                                            3*ones(par_size.T);
                                            3*ones(par_size.λ);
                                            3*ones(par_size.ρ)])

     θ_ini_bound = [ones(par_size.R);
                    zeros(par_size.d);
                    ones(par_size.Z + par_size.Z_plus);
                    -ones(par_size.Z_minus);
                    0.25*ones(par_size.Q);
                    -0.5*ones(par_size.Q_cov);
                    zeros(par_size.c);
                    0.5*ones(par_size.T);
                    (2*pi/32)*ones(par_size.λ);
                    0.5*ones(par_size.ρ)]

     θ_ini_unb = get_par_unb(θ_ini_bound, MIN, MAX, opt_transf)

     chain_θ_unb, chain_θ_bound, distr_α, distr_fcst, par, distr_par, mwg_const, acc_rate =
          mwg_run(θ_ini_unb, par, h, par_ind, par_size, prior_opt, MIN, MAX, opt_transf,
                  iter_init_adapt, iter_init_store, iter_main_adapt, iter_main_store,
                  copy(mwg_const), acc_target, adapt_interval, t, end_oos, σʸ)

     print("Main store > Final acceptance rate: $acc_rate%\n")

     return distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_size, distr_par
end

function mwg_main(par::ParSsm, h::Int64, nDraws::Array{Int64, 1}, burnin::Array{Int64, 1},
                  mwg_const::Array{Float64, 1}, par_ind::BoolParSsm, σʸ;
                  acc_target::Float64=0.25, adapt_interval::Int64=50, t=0::Int64, end_oos=0::Int64)

     iter_init_adapt = burnin[1]
     iter_init_store = nDraws[1] - burnin[1]
     iter_main_adapt = burnin[2]
     iter_main_store = nDraws[2] - burnin[2]

     return mwg_main(par, h, iter_init_adapt, iter_init_store, iter_main_adapt, iter_main_store,
                     mwg_const, par_ind, σʸ;
                     acc_target=acc_target, adapt_interval=adapt_interval, t=t, end_oos=end_oos)
end
