

# ----------------------------------------------------------------------------------------------------------------------
# Initial settings
# ----------------------------------------------------------------------------------------------------------------------
# choose model
model = "two_gap_AR2_6_obs" # "hasenzagl_2020", "kuttner_AR2_4_obs", "okun_kuttner_AR2_6_obs",  "two_gap_AR2_6_obs"

using Distributed;
using Base.Threads;
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

data_path = "./data/inflation_2025.xlsx"; # Data file
end_presample_vec = [31, 12, 1998]; # End presample, day/month/year [it is used when run_type is 2 or 3]
h = 8; # forecast horizon [it is used when run_type is 1 or 3]


# ----------------------------------------------------------------------------------------------------------------------
# Metropolis-Within-Gibbs settings
# ----------------------------------------------------------------------------------------------------------------------

nDraws    = [40_000; 40_000]; # [number of draws in initialization; number of draws in execusion]
burnin    = nDraws .- 20_000; # number of draws in the burn-in stage
mwg_const = [0.025; 0.25]; # Initial constant. mwg_const might be adjusted to get an acceptance rate between 25% and 35%




cond = [];

# Load data
data, date, nM, nQ, MNEMONIC = read_data(data_path, model);


# ----------------------------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------------------------

# This random seed gives a chain similar to the one computed in Julia 0.6.2 for the paper
Random.seed!(2);

# Run code
include("code/tc_main.jl");

display("Done!");

 
