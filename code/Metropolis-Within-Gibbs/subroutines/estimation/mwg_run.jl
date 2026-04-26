#=
This file is part of the replication code for: Hasenzagl, T., Pellegrino, F., Reichlin, L., & Ricco, G. (2020). A Model of the Fed's View on Inflation.
Please cite the paper if you are using any part of the code for academic work (including, but not limited to, conference and peer-reviewed papers).
=#

function update_adaptive_scale(scale::Float64, block_accept::Int64, block_draws::Int64, acc_target::Float64)
     if block_draws == 0
          return scale;
     end

     block_rate = block_accept / block_draws;
     return scale * exp(block_rate - acc_target);
end

function mwg_run(θ_ini_unb::Array{Float64,1}, par::ParSsm, h::Int64, par_ind::BoolParSsm, par_size::SizeParSsm,
                  prior_opt::PriorOpt, MIN::Array{Float64, 1}, MAX::Array{Float64, 1}, opt_transf::Array{Int64, 1},
                  iter_init_adapt::Int64, iter_init_store::Int64, iter_main_adapt::Int64, iter_main_store::Int64,
                  mwg_const::Array{Float64, 1}, acc_target::Float64, adapt_interval::Int64,
                  t::Int64, end_oos::Int64, σʸ)

     if length(mwg_const) != 2
          error("mwg_const must contain two proposal scales: one for init and one for main.");
     elseif iter_init_adapt < 0 || iter_main_adapt < 0
          error("Adaptation phase lengths must be non-negative.");
     elseif iter_init_store < 2
          error("iter_init_store must be at least 2 to build the empirical covariance matrix.");
     elseif iter_main_store < 1
          error("iter_main_store must be at least 1 to return a stored chain.");
     elseif adapt_interval < 1
          error("adapt_interval must be positive.");
     end

     k                = size(par.T)[1];
     n, m             = size(par.y);
     distr_α          = zeros(k, m, iter_main_store);
     distr_fcst       = zeros(h, n, iter_main_store);
     distr_par        = Array{Any}(undef, iter_main_store);
     chain_θ_unb      = zeros(par_size.θ, iter_main_store);
     chain_θ_bound    = zeros(par_size.θ, iter_main_store);
     chain_θ_unb_init = zeros(par_size.θ, iter_init_store);

     apriori_rejection = [0.0];
     par_prop          = copy(par);

     θ_unb   = copy(θ_ini_unb);
     θ_bound = get_par_bound(θ_unb, MIN, MAX, opt_transf);
     set_par!(θ_bound, θ_unb, par, opt_transf, MIN, MAX, par_ind, par_size, prior_opt, apriori_rejection, σʸ);

     if apriori_rejection[1] != 0 || !isfinite(par.logposterior)
          error("Initial parameter vector implies a non-finite posterior. Check the initialisation and bounds.");
     end

     init_scale         = mwg_const[1];
     main_scale         = mwg_const[2];
     I_Σ                = Matrix{Float64}(I, par_size.θ, par_size.θ);
     final_store_accept = falses(iter_main_store);

     function run_phase!(phase_name::String, n_draws::Int64, scale::Float64, Σ::Array{Float64, 2};
                         adapt_scale::Bool=false, store_chain_unb=nothing, store_chain_bound=nothing,
                         store_gibbs::Bool=false)

          if n_draws == 0
               return scale, falses(0)
          end

          L             = cholesky(Symmetric(Σ)).L;
          accept_phase  = falses(n_draws);
          block_accept  = 0;
          block_draws   = 0;
          current_scale = scale;

          for draw=1:n_draws
               last_draw = draw - 1;
               if draw > 1 && mod(last_draw, 1000) == 0
                    if t != 0 && end_oos != 0
                         print("OOS > Running the $t-th iteration (out of $end_oos) \n");
                    end

                    acc_rate = round(100 * mean(accept_phase[1:last_draw]), digits=2);
                    print("$phase_name > Executed $last_draw draws (out of $n_draws): \n");
                    print("- Logposterior: $(round(par.logposterior, digits=2))\n");
                    print("- Acceptance rate: $acc_rate%\n\n");
               end

               θ_prop_unb   = θ_unb + current_scale .* (L * randn(par_size.θ));
               θ_prop_bound = get_par_bound(θ_prop_unb, MIN, MAX, opt_transf);

               apriori_rejection[1] = 0.0;
               par_prop = copy(par);
               set_par!(θ_prop_bound, θ_prop_unb, par_prop, opt_transf, MIN, MAX, par_ind, par_size, prior_opt, apriori_rejection, σʸ);

               accepted = false;
               if apriori_rejection[1] == 0
                    log_accept = min(0.0, par_prop.logposterior - par.logposterior);
                    accepted   = log(rand()) < log_accept;

                    if accepted
                         par     = copy(par_prop);
                         θ_unb   = copy(θ_prop_unb);
                         θ_bound = copy(θ_prop_bound);
                    end
               end

               accept_phase[draw] = accepted;
               block_draws += 1;
               if accepted
                    block_accept += 1;
               end

               if store_chain_unb !== nothing
                    store_chain_unb[:, draw] = θ_unb;
               end

               if store_chain_bound !== nothing
                    store_chain_bound[:, draw] = θ_bound;
               end

                if store_gibbs
                     α_draw, _               = kalman_diffuse!(par, 0, 1, 1);
                     distr_α[:, :, draw]     = α_draw;
                     distr_par[draw]         = copy(par);
                    density_fcst_draw       = (par.Z * α_draw)';
                    density_fcst_draw       = density_fcst_draw[end-h+1:end, :];
                    distr_fcst[:, :, draw]  = density_fcst_draw;
                end

                if adapt_scale && mod(draw, adapt_interval) == 0
                     current_scale = update_adaptive_scale(current_scale, block_accept, block_draws, acc_target);
                     block_accept  = 0;
                     block_draws   = 0;
                end
           end

           if adapt_scale && block_draws > 0
                current_scale = update_adaptive_scale(current_scale, block_accept, block_draws, acc_target);
           end

           phase_acc_rate = round(100 * mean(accept_phase), digits=2);
           print("$phase_name > Acceptance rate: $phase_acc_rate%\n");

          return current_scale, accept_phase
     end

     init_scale, _ = run_phase!("Init adapt", iter_init_adapt, init_scale, I_Σ; adapt_scale=true);

     _, _ = run_phase!("Init store", iter_init_store, init_scale, I_Σ;
                       store_chain_unb=chain_θ_unb_init);

     θ_start_unb = median(permutedims(chain_θ_unb_init), dims=1)[:];
     Σ_mwg       = cov(permutedims(chain_θ_unb_init));
     Σ_mwg      += 1e-10 .* Matrix{Float64}(I, par_size.θ, par_size.θ);

     θ_bound = get_par_bound(θ_start_unb, MIN, MAX, opt_transf);
     apriori_rejection[1] = 0.0;
     set_par!(θ_bound, θ_start_unb, par, opt_transf, MIN, MAX, par_ind, par_size, prior_opt, apriori_rejection, σʸ);
     if apriori_rejection[1] != 0 || !isfinite(par.logposterior)
          error("Median init-store draw implies a non-finite posterior.");
     end
     θ_unb = copy(θ_start_unb);

     main_scale, _ = run_phase!("Main adapt", iter_main_adapt, main_scale, Σ_mwg; adapt_scale=true);

     _, final_store_accept = run_phase!("Main store", iter_main_store, main_scale, Σ_mwg;
                                        store_chain_unb=chain_θ_unb,
                                        store_chain_bound=chain_θ_bound,
                                        store_gibbs=true);

     acc_rate        = round(100 * mean(final_store_accept), digits=2);
     tuned_mwg_const = [init_scale; main_scale];

     return chain_θ_unb, chain_θ_bound, distr_α, distr_fcst, par, distr_par, tuned_mwg_const, acc_rate;
end
