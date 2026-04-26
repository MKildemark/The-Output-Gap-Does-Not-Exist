

# ----------------------------------------------------------------------------------------------------------------------
# Initial settings
# ----------------------------------------------------------------------------------------------------------------------
# choose model
model = "hasenzagl_2020" # "two_gap_AR2_6_obs", "two_gap_AR2_6_obs_lags", "kuttner_AR2_4_obs", "okun_kuttner_AR2_6_obs", "hasenzagl_2020"

using Distributed;

include("code/read_data.jl");
include("code/tc_models/tc_$(model).jl");

@everywhere include("code/Metropolis-Within-Gibbs/MetropolisWithinGibbs.jl")
@everywhere using DataFrames, Dates, FileIO, JLD2, LinearAlgebra, Random, Statistics, XLSX;
@everywhere using Main.MetropolisWithinGibbs;

# 1. Single iteration: it executes the code using the most updated datapoints
# 2. Conditional forecast (you need to run option 1 first)
# 3. Out-of-sample: out-of-sample exercise, forecasting period starts after end_presample_vec
run_type = 1;

if run_type == 1
	res_name = "$(model)_iis"
elseif run_type == 2
	res_name = "$(model)_cond"
elseif run_type == 3
	res_name = "$(model)_oos"
end
res_name_iis = "$(model)_iis";

data_path = "./data/inflation_2025.xlsx";
end_presample_vec = [31, 12, 1998];
h = 8;



# ----------------------------------------------------------------------------------------------------------------------
# Metropolis-Within-Gibbs settings
# ----------------------------------------------------------------------------------------------------------------------

iter_init_adapt = 20000;
iter_init_store = 20000;
iter_main_adapt = 20000;
iter_main_store = 20000;
mwg_const       = [0.025; 0.25];
acc_target      = 0.25;
adapt_interval  = 100;

# Derived legacy metadata kept for compatibility with existing result readers.
nDraws = [iter_init_adapt + iter_init_store; iter_main_adapt + iter_main_store];
burnin = [iter_init_adapt; iter_main_adapt];



cond = [];

data, date, nM, nQ, MNEMONIC = read_data(data_path, model);


# ----------------------------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------------------------

Random.seed!(2);

include("code/tc_main.jl");

display("Done!");
