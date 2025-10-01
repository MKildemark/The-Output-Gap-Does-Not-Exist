

# ----------------------------------------------------------------------------------------------------------------------
# Initial settings
# ----------------------------------------------------------------------------------------------------------------------
# choose model
model = "full" # "full", "inflation", "employment", "employment+inflation", "flex", "efficient", "full_inf", "full_no_lags"

using Distributed;
include("code/read_data.jl");
if model == "full"
        include("code/tc_models/tc_full_model.jl");

elseif model == "full_no_lags"
        include("code/tc_models/tc_full_no_lags.jl")
        
elseif model == "inflation"
        include("code/tc_models/tc_inflation_model.jl");

elseif model == "employment"
        include("code/tc_models/tc_employment_model.jl");

elseif model == "employment+inflation"
        include("code/tc_models/tc_employment_inflation_model.jl");
  
elseif model == "flex"
        include("code/tc_models/tc_flex_gap.jl");

elseif model == "efficient"
        include("code/tc_models/tc_efficient_gap.jl");
      
elseif model == "full_inf"
        include("code/tc_models/tc_full_model_inf.jl");
       
else
    error("Model not recognized. Choose either 'full', 'inflation', 'employment' or 'employment+inflation'.")
end

@everywhere include("code/Metropolis-Within-Gibbs/MetropolisWithinGibbs.jl")
@everywhere using DataFrames, Dates, FileIO, JLD2, LinearAlgebra, Random, Statistics, XLSX;
@everywhere using Main.MetropolisWithinGibbs;

res_name = model
data_path = "./data/inflation.xlsx"; # Data file
end_presample_vec = [31, 12, 1998]; # End presample, day/month/year [it is used when run_type is 2 or 3]
h = 0; # forecast horizon [it is used when run_type is 1 or 3]


# ----------------------------------------------------------------------------------------------------------------------
# Metropolis-Within-Gibbs settings
# ----------------------------------------------------------------------------------------------------------------------

nDraws    = [40000; 40000]; # [number of draws in initialization; number of draws in execusion]
burnin    = nDraws .- 20000; # number of draws in the burn-in stage
mwg_const = [0.025; 0.25]; # Initial constant. mwg_const might be adjusted to get an acceptance rate between 25% and 35%



run_type = 1; # 1: single iteration; 2: out-of-sample; 3: conditional forecasts
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

 
