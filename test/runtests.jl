using Test
using Random
using Dates
using FileIO
using JLD2
using LinearAlgebra
using Statistics
using DataFrames
using XLSX

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_PATH = joinpath(ROOT, "data", "inflation_2025.xlsx")
const SMOKE_SETTINGS = (
    h = 2,
    iter_init_adapt = 5,
    iter_init_store = 5,
    iter_main_adapt = 5,
    iter_main_store = 4,
    mwg_const = [0.025; 0.25],
    acc_target = 0.25,
    adapt_interval = 4,
)

include(joinpath(ROOT, "code", "Metropolis-Within-Gibbs", "MetropolisWithinGibbs.jl"))
using Main.MetropolisWithinGibbs

include(joinpath(ROOT, "code", "read_data.jl"))
run_type = 0
include(joinpath(ROOT, "code", "tc_main.jl"))

function refresh_tc_main()
    global run_type = 0
    include(joinpath(ROOT, "code", "tc_main.jl"))
end

function include_model(model_file::String)
    include(joinpath(ROOT, "code", "tc_models", model_file))
end

function load_model_data(model_name::String; rows::UnitRange{Int}=1:36)
    data, date, nM, nQ, MNEMONIC = read_data(DATA_PATH, model_name)
    return copy(data[rows, :]), date[rows], nM, nQ, MNEMONIC
end

function expected_original_forecast(distr_par, distr_α, σʸ, h, draw)
    density_fcst_draw = (distr_par[draw].Z * distr_α[:, :, draw])'
    return density_fcst_draw[end-h+1:end, :] .* σʸ
end

function run_iis_smoke(tmpdir::String, model_name::String, model_file::String)
    include_model(model_file)
    refresh_tc_main()
    data, date, nM, nQ, MNEMONIC = load_model_data(model_name)
    res_name = "$(model_name)_iis_smoke"

    cd(tmpdir) do
        Random.seed!(1)
        Base.invokelatest(tc_iis_run, copy(data), date, nM, nQ, MNEMONIC, SMOKE_SETTINGS.h,
                          SMOKE_SETTINGS.iter_init_adapt, SMOKE_SETTINGS.iter_init_store,
                          SMOKE_SETTINGS.iter_main_adapt, SMOKE_SETTINGS.iter_main_store,
                          copy(SMOKE_SETTINGS.mwg_const), SMOKE_SETTINGS.acc_target,
                          SMOKE_SETTINGS.adapt_interval, res_name)
        return load("res_$(res_name).jld2")
    end
end

@testset "ParSsm copy isolation" begin
    y = Union{Float64, Missing}[1.0  missing; 2.0  3.0]
    par = ParSsm(copy(y),
                 [1.0, 2.0],
                 [1.0 0.0; 0.0 1.0],
                 [0.1 0.0; 0.0 0.1],
                 [0.0, 0.0],
                 [1.0 0.0; 0.0 1.0],
                 [0.2 0.0; 0.0 0.2],
                 [0.0, 0.0],
                 zeros(2, 2),
                 zeros(2, 2),
                 [0.5, 0.0],
                 [0.8, 0.0],
                 0.0, 0.0, 0.0)

    par_clone = copy(par)

    @test par_clone !== par
    @test par_clone.y !== par.y
    @test par_clone.d !== par.d
    @test par_clone.Z !== par.Z
    @test par_clone.Q !== par.Q
    @test par_clone.λ !== par.λ

    par_clone.d[1] = 99.0
    par_clone.Z[1, 1] = 77.0
    par_clone.y[1, 2] = 4.0

    @test par.d[1] == 1.0
    @test par.Z[1, 1] == 1.0
    @test ismissing(par.y[1, 2])
end

@testset "Hasenzagl data loader returns 8 observables" begin
    data, _, nM, nQ, MNEMONIC = read_data(DATA_PATH, "hasenzagl_2020")

    @test nM == 0
    @test nQ == 8
    @test size(data, 2) == 8
    @test MNEMONIC == ["GDP", "EMPL", "U", "OIL", "INFL", "CORE", "UOM", "SPF"]
end

@testset "Adaptive scaling helper handles trailing blocks" begin
    full_block_scale = MetropolisWithinGibbs.update_adaptive_scale(1.0, 2, 4, 0.25)
    trailing_scale = MetropolisWithinGibbs.update_adaptive_scale(full_block_scale, 0, 1, 0.25)

    @test trailing_scale != full_block_scale
    @test trailing_scale ≈ full_block_scale * exp(-0.25)
end

@testset "IIS smoke runs save corrected metadata" begin
    mktempdir() do tmpdir
        models = [
            ("two_gap_AR2_6_obs", "tc_two_gap_AR2_6_obs.jl"),
            ("two_gap_AR2_6_obs_lags", "tc_two_gap_AR2_6_obs_lags.jl"),
            ("kuttner_AR2_4_obs", "tc_kuttner_AR2_4_obs.jl"),
            ("okun_kuttner_AR2_6_obs", "tc_okun_kuttner_AR2_6_obs.jl"),
            ("hasenzagl_2020", "tc_hasenzagl_2020.jl"),
        ]

        for (model_name, model_file) in models
            res = run_iis_smoke(tmpdir, model_name, model_file)

            @test res["iter_main_store"] == SMOKE_SETTINGS.iter_main_store
            @test size(res["chain_θ_bound"], 2) == SMOKE_SETTINGS.iter_main_store
            @test length(res["distr_par"]) == SMOKE_SETTINGS.iter_main_store
            @test size(res["distr_fcst"], 3) == SMOKE_SETTINGS.iter_main_store
            @test res["forecast_scale"] == "original_units"
            @test res["distr_par"][1].Q !== res["distr_par"][2].Q

            expected_fcst = expected_original_forecast(res["distr_par"], res["distr_α"], res["σʸ"], SMOKE_SETTINGS.h, 1)
            @test res["distr_fcst"][:, :, 1] ≈ expected_fcst
        end
    end
end

@testset "Conditional and OOS smoke runs keep original-unit forecasts" begin
    mktempdir() do tmpdir
        include_model("tc_two_gap_AR2_6_obs.jl")
        refresh_tc_main()
        data, date, nM, nQ, MNEMONIC = load_model_data("two_gap_AR2_6_obs"; rows=1:24)
        res_name = "two_gap_cond_base"
        end_presample_vec = [day(date[end-1]), month(date[end-1]), year(date[end-1])]

        cd(tmpdir) do
            Random.seed!(2)
            Base.invokelatest(tc_iis_run, copy(data), date, nM, nQ, MNEMONIC, SMOKE_SETTINGS.h,
                              SMOKE_SETTINGS.iter_init_adapt, SMOKE_SETTINGS.iter_init_store,
                              SMOKE_SETTINGS.iter_main_adapt, SMOKE_SETTINGS.iter_main_store,
                              copy(SMOKE_SETTINGS.mwg_const), SMOKE_SETTINGS.acc_target,
                              SMOKE_SETTINGS.adapt_interval, res_name)

            cond = Any[Dict(MNEMONIC[1] => [0.0, 0.0])]
            Base.invokelatest(tc_cond_fc_run, copy(data), date, nM, nQ, MNEMONIC, SMOKE_SETTINGS.h, "two_gap_cond", cond, res_name)
            Base.invokelatest(tc_oos_run, copy(data), date, nM, nQ, MNEMONIC, SMOKE_SETTINGS.h,
                              SMOKE_SETTINGS.iter_init_adapt, SMOKE_SETTINGS.iter_init_store,
                              SMOKE_SETTINGS.iter_main_adapt, SMOKE_SETTINGS.iter_main_store,
                              copy(SMOKE_SETTINGS.mwg_const), SMOKE_SETTINGS.acc_target,
                              SMOKE_SETTINGS.adapt_interval, "two_gap_oos", end_presample_vec)

            res_iis = load("res_$(res_name).jld2")
            res_cond = load("res_two_gap_cond_cond1.jld2")
            res_oos = load("res_two_gap_oos_chunk1.jld2")

            @test res_cond["forecast_scale"] == "original_units"
            @test res_oos["forecast_scale"] == "original_units"
            @test size(res_oos["chain_θ_bound"], 2) == SMOKE_SETTINGS.iter_main_store

            par_draw = copy(res_iis["distr_par"][1])
            par_draw.y = permutedims(res_cond["data_ith"])
            α_draw, _ = kalman_diffuse!(par_draw, 0, 1, 1)
            @test isapprox(res_cond["distr_fcst_cond"][end-SMOKE_SETTINGS.h+1:end, :, 1],
                           ((par_draw.Z * α_draw)' .* res_iis["σʸ"])[end-SMOKE_SETTINGS.h+1:end, :];
                           rtol=1e-2)

            expected_oos_fcst = expected_original_forecast(res_oos["distr_par"], res_oos["distr_α"], res_oos["σʸ"], SMOKE_SETTINGS.h, 1)
            @test res_oos["distr_fcst"][:, :, 1] ≈ expected_oos_fcst
        end
    end
end
