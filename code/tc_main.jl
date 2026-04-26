#=
This file is part of the replication code for: Hasenzagl, T., Pellegrino, F., Reichlin, L., & Ricco, G. (2020). A Model of the Fed's View on Inflation.
Please cite the paper if you are using any part of the code for academic work (including, but not limited to, conference and peer-reviewed papers).
=#

#=
Name: tc_main.jl
Description: Execution manager
=#


#=
--------------------------------------------------------------------------------------------------
Dependencies
--------------------------------------------------------------------------------------------------
=#

"""
    standardise_data!(data::Matrix{Union{Float64, Missing}}, nM::Int64, nQ::Int64)

Standardise data.
"""
function standardise_data!(data::Matrix{Union{Float64, Missing}}, nM::Int64, nQ::Int64)

    if nM > 0 && nQ > 0
        σʸ_monthly   = hcat([std(collect(skipmissing(diff(data[:, i]))), dims=1) for i=1:nM]...);
        σʸ_quarterly = hcat([std(collect(skipmissing(diff(data[3:3:end, i]))), dims=1) for i=nM+1:nM+nQ]...);
        σʸ           = [σʸ_monthly σʸ_quarterly];
    else
        σʸ = hcat([std(collect(skipmissing(diff(data[:, i]))), dims=1) for i=1:nM+nQ]...);
    end

    data ./= σʸ;
    return σʸ;
end

function get_legacy_draw_settings(iter_init_adapt::Int64, iter_init_store::Int64,
                                  iter_main_adapt::Int64, iter_main_store::Int64)

    nDraws = [iter_init_adapt + iter_init_store; iter_main_adapt + iter_main_store];
    burnin = [iter_init_adapt; iter_main_adapt];
    return nDraws, burnin;
end

function get_iter_main_store(res_iis, distr_par)
    if haskey(res_iis, "iter_main_store")
        return res_iis["iter_main_store"];
    elseif haskey(res_iis, "chain_θ_bound")
        return size(res_iis["chain_θ_bound"], 2);
    else
        return length(distr_par);
    end
end

function rescale_forecasts_to_original!(distr_fcst, σʸ)
    for draw in axes(distr_fcst, 3)
        distr_fcst[:, :, draw] .= distr_fcst[:, :, draw] .* σʸ;
    end

    return distr_fcst;
end

"""
    tc_iis_run(data::Matrix{Union{Float64, Missing}}, date::Vector{Date}, nM::Int64, nQ::Int64,
               MNEMONIC::Vector{String}, h::Int64, iter_init_adapt::Int64, iter_init_store::Int64,
               iter_main_adapt::Int64, iter_main_store::Int64, mwg_const::Vector{Float64},
               acc_target::Float64, adapt_interval::Int64, res_name::String)

Single iteration: it executes the code using the most updated datapoints.
"""
function tc_iis_run(data::Matrix{Union{Float64, Missing}}, date::Vector{Date}, nM::Int64, nQ::Int64,
                    MNEMONIC::Vector{String}, h::Int64, iter_init_adapt::Int64, iter_init_store::Int64,
                    iter_main_adapt::Int64, iter_main_store::Int64, mwg_const::Vector{Float64},
                    acc_target::Float64, adapt_interval::Int64, res_name::String)

    σʸ = standardise_data!(data, nM, nQ);
    data = [data; missing.*ones(h, nM+nQ)];

    nDraws, burnin = get_legacy_draw_settings(iter_init_adapt, iter_init_store, iter_main_adapt, iter_main_store);

    distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_ind, par_size, distr_par =
         tc_mwg(data, h, iter_init_adapt, iter_init_store, iter_main_adapt, iter_main_store,
                mwg_const, σʸ; acc_target=acc_target, adapt_interval=adapt_interval);

    rescale_forecasts_to_original!(distr_fcst, σʸ);

    save(string("./res_", res_name, ".jld2"), Dict("distr_α" => distr_α, "distr_fcst" => distr_fcst, "chain_θ_unb" => chain_θ_unb,
                "chain_θ_bound" => chain_θ_bound, "mwg_const" => mwg_const, "acc_rate" => acc_rate, "par" => par,
                "iter_init_adapt" => iter_init_adapt, "iter_init_store" => iter_init_store,
                "iter_main_adapt" => iter_main_adapt, "iter_main_store" => iter_main_store,
                "acc_target" => acc_target, "adapt_interval" => adapt_interval,
                "forecast_scale" => "original_units",
                "nDraws" => nDraws, "burnin" => burnin, "data" => data, "date" => date, "nM" => nM, "nQ" => nQ,
                "MNEMONIC" => MNEMONIC, "par_ind" => par_ind, "par_size" => par_size, "distr_par" => distr_par, "σʸ" => σʸ));
end

"""
    tc_cond_fc_run(data::Matrix{Union{Float64, Missing}}, date::Vector{Date}, nM::Int64, nQ::Int64,
                   MNEMONIC::Vector{String}, h::Int64, res_name::String, cond::Vector{Any},
                   res_name_iis::String)

Conditional forecasts.
"""
function tc_cond_fc_run(data::Matrix{Union{Float64, Missing}}, date::Vector{Date}, nM::Int64, nQ::Int64,
                        MNEMONIC::Vector{String}, h::Int64, res_name::String, cond::Vector{Any},
                        res_name_iis::String)

    res_iis = load(string("./res_", res_name_iis, ".jld2"));

    data            = res_iis["data"];
    date            = res_iis["date"];
    σʸ              = res_iis["σʸ"];
    distr_par       = res_iis["distr_par"];
    MNEMONIC        = res_iis["MNEMONIC"];
    iter_main_store = get_iter_main_store(res_iis, distr_par);

    data = data[1:end-(size(data)[1] - size(date)[1]), :];

    for i=1:length(cond)

        observations, n = size(data);

        keys_ith   = collect(keys(cond[i]));
        values_ith = collect(values(cond[i]));
        pos_ith    = vcat([findall(MNEMONIC .== j) for j in keys_ith]...);
        h_ith      = length.(values_ith);

        data_ith = copy(data);
        data_ith = [data; missing .* ones(maximum([h; h_ith]), n)];
        for j=1:length(pos_ith)
            data_ith[observations+1:observations+h_ith[j], pos_ith[j]] = values_ith[j] ./ σʸ[pos_ith[j]];
        end

        k               = size(distr_par[1].T)[1];
        m, n            = size(data_ith);
        distr_fcst_cond = zeros(m, n, iter_main_store);
        distr_α_cond    = zeros(k, m, iter_main_store);

        for draw=1:iter_main_store
            if draw > 1 && mod(draw, 100) == 0
                print("Conditional forecast $i (out of $(size(cond)[1])) > $draw-th iteration (out of $iter_main_store) \n");
            end

            par_draw                 = copy(distr_par[draw]);
            par_draw.y               = permutedims(data_ith);
            α_draw, _                = kalman_diffuse!(par_draw, 0, 1, 1);
            distr_α_cond[:, :, draw] = α_draw;
            distr_fcst_cond[:, :, draw] = (par_draw.Z * α_draw)' .* σʸ;
        end

        print("\n");

        save(string("./res_", res_name, "_cond$(i).jld2"), Dict("data_ith" => data_ith, "distr_fcst_cond" => distr_fcst_cond,
                    "distr_α_cond" => distr_α_cond, "conditional_path" => cond[i], "σʸ" => σʸ,
                    "forecast_scale" => "original_units"));
    end
end

"""
    tc_oos_run(data::Matrix{Union{Float64, Missing}}, date::Vector{Date}, nM::Int64, nQ::Int64,
               MNEMONIC::Vector{String}, h::Int64, iter_init_adapt::Int64, iter_init_store::Int64,
               iter_main_adapt::Int64, iter_main_store::Int64, mwg_const::Vector{Float64},
               acc_target::Float64, adapt_interval::Int64, res_name::String,
               end_presample_vec::Vector{Int64})

Out-of-sample: out-of-sample exercise, forecasting period starts after end_presample_vec.
"""
function tc_oos_run(data::Matrix{Union{Float64, Missing}}, date::Vector{Date}, nM::Int64, nQ::Int64,
                    MNEMONIC::Vector{String}, h::Int64, iter_init_adapt::Int64, iter_init_store::Int64,
                    iter_main_adapt::Int64, iter_main_store::Int64, mwg_const::Vector{Float64},
                    acc_target::Float64, adapt_interval::Int64, res_name::String,
                    end_presample_vec::Vector{Int64})

    data_full = copy(data);
    nDraws, burnin = get_legacy_draw_settings(iter_init_adapt, iter_init_store, iter_main_adapt, iter_main_store);

    date_vec      = vcat([[Dates.day(date[i]) Dates.month(date[i]) Dates.year(date[i])] for i=1:length(date)]...);
    end_presample = findall(sum(date_vec .== end_presample_vec', dims=2)[:] .== 3)[1];
    end_oos       = size(date_vec)[1];

    if end_presample == end_oos
        error("end_presample_vec is not set correctly");
    end

    oos_length = end_oos-end_presample+1;

    for t=end_presample:end_oos
        print("Out-of-sample iteration > ", t-end_presample+1, " (out of ", oos_length, ")\n");

        data_t = data_full[1:t, :];
        σʸ = standardise_data!(data_t, nM, nQ);
        data_t = [data_t; missing.*ones(h, nM+nQ)];

        distr_α, distr_fcst, chain_θ_unb, chain_θ_bound, mwg_const, acc_rate, par, par_ind, par_size, distr_par =
             tc_mwg(data_t, h, iter_init_adapt, iter_init_store, iter_main_adapt, iter_main_store,
                    mwg_const, σʸ; acc_target=acc_target, adapt_interval=adapt_interval,
                    t=t-end_presample+1, end_oos=oos_length);

        rescale_forecasts_to_original!(distr_fcst, σʸ);

        save(string("./res_", res_name, "_chunk", t-end_presample+1, ".jld2"), Dict("distr_α" => distr_α, "distr_fcst" => distr_fcst,
                    "chain_θ_unb" => chain_θ_unb, "chain_θ_bound" => chain_θ_bound, "mwg_const" => mwg_const,
                    "acc_rate" => acc_rate, "par" => par, "par_ind" => par_ind, "par_size" => par_size,
                    "iter_init_adapt" => iter_init_adapt, "iter_init_store" => iter_init_store,
                    "iter_main_adapt" => iter_main_adapt, "iter_main_store" => iter_main_store,
                    "acc_target" => acc_target, "adapt_interval" => adapt_interval,
                    "forecast_scale" => "original_units",
                    "nDraws" => nDraws, "burnin" => burnin, "distr_par" => distr_par, "data" => data_t, "σʸ" => σʸ));
    end

    save(string("./res_", res_name, "_chunk0.jld2"), Dict("end_presample" => end_presample, "end_oos" => end_oos,
                "oos_length" => oos_length, "iter_init_adapt" => iter_init_adapt,
                "iter_init_store" => iter_init_store, "iter_main_adapt" => iter_main_adapt,
                "iter_main_store" => iter_main_store, "acc_target" => acc_target,
                "adapt_interval" => adapt_interval, "forecast_scale" => "original_units",
                "nDraws" => nDraws, "burnin" => burnin, "date" => date,
                "nM" => nM, "nQ" => nQ, "MNEMONIC" => MNEMONIC, "data_full" => data_full));
end


#=
--------------------------------------------------------------------------------------------------
Execution manager
--------------------------------------------------------------------------------------------------
=#

if run_type == 1
    tc_iis_run(data, date, nM, nQ, MNEMONIC, h, iter_init_adapt, iter_init_store,
               iter_main_adapt, iter_main_store, mwg_const, acc_target, adapt_interval, res_name);
elseif run_type == 2
    tc_cond_fc_run(data, date, nM, nQ, MNEMONIC, h, res_name, cond, res_name_iis);
elseif run_type == 3
    tc_oos_run(data, date, nM, nQ, MNEMONIC, h, iter_init_adapt, iter_init_store,
               iter_main_adapt, iter_main_store, mwg_const, acc_target, adapt_interval,
               res_name, end_presample_vec);
end
