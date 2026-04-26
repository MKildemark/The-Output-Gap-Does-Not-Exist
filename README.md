# Estimating Output Gaps: Repository Guide

This repository estimates trend-cycle state-space models for output gaps and inflation components using a custom Julia Metropolis-within-Gibbs sampler. The main workflow is:

1. choose a model and run configuration in `user_main.jl`;
2. load and standardize data with `code/read_data.jl` and `code/tc_main.jl`;
3. construct the chosen state-space model from `code/tc_models/`;
4. estimate parameters and latent states with `code/Metropolis-Within-Gibbs/`;
5. save `.jld2` results;
6. use the result notebooks to build figures and decompositions.

The code is research code, not a package-style application. A future agent should prioritize preserving model semantics and saved-result compatibility over broad refactors.

## Top-Level Files

- `user_main.jl`: primary execution script. Selects `model`, `run_type`, input data path, forecast horizon, sampler iteration counts, initial proposal scales, and adaptation controls.
- `Project.toml` / `Manifest.toml`: Julia environment.
- `res_*.jld2`: saved estimation results. These are large binary outputs consumed by notebooks.
- `results_*.ipynb`: analysis and figure notebooks.
- `data/`: input datasets.
- `csv_output/`: exported CSV outputs from analysis notebooks.
- `Results/figs/`: generated figures.
- `code/`: all model, data, and sampler source code.

## Running The Main Estimation

Open `user_main.jl` and set:

- `model`
  - `two_gap_AR2_6_obs`: baseline empirical two-gap model.
  - `two_gap_AR2_6_obs_lags`: exploratory two-gap model with extra lagged common loadings.
  - `kuttner_AR2_4_obs`: 4-observable one-gap benchmark.
  - `okun_kuttner_AR2_6_obs`: 6-observable one-gap benchmark.
  - `hasenzagl_2020`: Hasenzagl-Pellegrino-Reichlin-Ricco style inflation model.
- `run_type`
  - `1`: in-sample estimation, saves `res_<model>_iis.jld2`.
  - `2`: conditional forecasts, requires an existing `res_<model>_iis.jld2`.
  - `3`: out-of-sample exercise, saves `res_<model>_oos_chunk*.jld2`.
- sampler settings:
  - `iter_init_adapt`
  - `iter_init_store`
  - `iter_main_adapt`
  - `iter_main_store`
  - `mwg_const`
  - `acc_target`
  - `adapt_interval`

`nDraws` and `burnin` are derived legacy metadata:

```julia
nDraws = [iter_init_adapt + iter_init_store; iter_main_adapt + iter_main_store]
burnin = [iter_init_adapt; iter_main_adapt]
```

Keep this shape unless all result readers are updated. The notebooks and `tc_main.jl` expect `nDraws[end] - burnin[end]` to equal the final stored posterior draw count for new results.

## Execution Manager: `code/tc_main.jl`

`tc_main.jl` defines the run modes used by `user_main.jl`.

- `standardise_data!`: standardizes monthly and quarterly observables by the standard deviation of first differences.
- `tc_iis_run`: standardizes full data, appends forecast-horizon missing observations, runs `tc_mwg`, and saves `res_<model>_iis.jld2`.
- `tc_cond_fc_run`: loads an in-sample result and runs conditional forecasts over stored posterior draws. It uses `length(distr_par)` from the loaded result, not the caller's current draw settings.
- `tc_oos_run`: repeatedly truncates the sample, re-estimates the model, rescales forecasts, and saves OOS chunks.

Saved in-sample result keys include:

- `distr_α`: sampled latent states, dimensions `(state, time, draw)`.
- `distr_fcst`: sampled forecasts, dimensions `(horizon, observable, draw)`.
- `chain_θ_unb`: final stored unconstrained parameter draws.
- `chain_θ_bound`: final stored bounded/natural parameter draws.
- `distr_par`: stored `ParSsm` objects for posterior draws.
- `mwg_const`: tuned proposal scales `[init_scale; main_scale]`.
- `acc_rate`: final main-store acceptance rate, scalar percent.
- `nDraws`, `burnin`: legacy draw metadata.
- `data`, `date`, `nM`, `nQ`, `MNEMONIC`, `par_ind`, `par_size`, `σʸ`.

## Model Wrappers: `code/tc_models/`

Every model file defines:

```julia
function tc_mwg(y, h, nDraws, burnin, mwg_const, σʸ; acc_target=0.25, adapt_interval=50)
```

The wrapper builds:

- observation equation pieces: `d`, `Z`, `R`;
- transition equation pieces: `c`, `T`, `Q`;
- diffuse and stationary initial conditions: `α¹`, `P¹`, `P̄¹`;
- trigonometric-cycle markers: `λ`, `ρ`;
- boolean masks in `BoolParSsm` saying which entries are estimated.

Then it calls `mwg_main` and returns:

```julia
distr_α, distr_fcst, chain_θ_unb, chain_θ_bound,
mwg_const, acc_rate, par, par_ind, par_size, distr_par
```

The parameter order in `mwg_main.jl`, `set_par.jl`, and the model masks must stay synchronized:

1. `R`
2. `d`
3. unrestricted `Z`
4. positive-restricted `Z_plus`
5. negative-restricted `Z_minus`
6. `Q`
7. `Q_cov`
8. `c`
9. `T`
10. `λ`
11. `ρ`

Do not reorder masks in a model wrapper without also updating the bounds, transformations, priors, and `set_par!`.

## Sampler Module

The module lives in `code/Metropolis-Within-Gibbs/`.

### `MetropolisWithinGibbs.jl`

Defines core types:

- `ParSsm`: mutable state-space model object.
- `SizeParSsm`: counts of estimated parameters by block.
- `BoolParSsm`: boolean masks for estimated parameters.
- `PriorOpt`: prior distribution objects/constants.

Important copy behavior:

- `copy(::ParSsm)` must copy array fields, not just the struct shell. This avoids proposal mutations contaminating the current chain state.
- `copy(::BoolParSsm)` copies mask arrays.
- `copy(::SizeParSsm)` and `copy(::PriorOpt)` preserve value fields.

Do not replace these with a generic shallow struct copy.

### `subroutines/estimation/mwg_main.jl`

Builds bounds, priors, transformations, initial parameter values, and calls `mwg_run`.

It has two APIs:

```julia
mwg_main(par, h,
         iter_init_adapt, iter_init_store, iter_main_adapt, iter_main_store,
         mwg_const, par_ind, σʸ; acc_target, adapt_interval, t, end_oos)
```

and the legacy metadata wrapper:

```julia
mwg_main(par, h, nDraws, burnin, mwg_const, par_ind, σʸ;
         acc_target, adapt_interval, t, end_oos)
```

The wrapper derives:

```julia
iter_init_adapt = burnin[1]
iter_init_store = nDraws[1] - burnin[1]
iter_main_adapt = burnin[2]
iter_main_store = nDraws[2] - burnin[2]
```

### `subroutines/estimation/mwg_run.jl`

Runs the four logical phases:

1. `Init adapt`: identity proposal shape, adaptive scale.
2. `Init store`: identity proposal shape, fixed scale, stores parameter draws for empirical covariance.
3. `Main adapt`: empirical covariance proposal shape, adaptive scale.
4. `Main store`: empirical covariance proposal shape, fixed scale, stores posterior parameter/state/forecast draws.

Important details:

- `acc_target` is a probability, e.g. `0.25`, not a percent like `25.0`.
- `mwg_const` must be finite and strictly positive.
- proposals with non-finite posterior are rejected.
- proposal covariance factorization uses `proposal_cholesky`, which tries increasing diagonal jitter before failing.
- returned chains contain only final stored posterior draws, not the main adaptation phase.

### Other Sampler Files

- `kalman_diffuse!.jl`: Kalman filtering/smoothing/simulation smoother.
- `set/set_par.jl`: maps sampled parameters into a `ParSsm`, evaluates priors/Jacobian/likelihood.
- `get/get_par_bound.jl`: maps unconstrained parameters to bounded/natural scale.
- `get/get_par_unb.jl`: inverse transformation.
- `get/get_logjacobian.jl`: Jacobian adjustment.
- `get/get_mwg_jump.jl`: legacy proposal-jump helper; uses the same robust Cholesky helper.
- `get/get_progress.jl`: legacy acceptance-rate helper. It infers acceptance from changes in the first parameter and is not reliable for new sampler diagnostics.
- `extra/`: small matrix utilities.

## Two-Gap Measurement Rewrites

`set_par.jl` contains local extensions for the two-gap models:

- `is_two_gap_baseline_model`
- `is_two_gap_lag_model`
- `rewrite_two_gap_baseline_common_block!`
- `rewrite_two_gap_lag_common_block!`

These functions take raw sampled common-loading parameters and rewrite the first columns of `Z` to impose the model's structural loading restrictions.

Important invariant:

- Priors for `Z` and `Z_plus` must be evaluated on the raw sampled parameters, before composite rewrites. The current code stores raw `θ_Z` / `θ_Z_plus` for prior evaluation and only then rewrites `par.Z`.

## Known Modeling Assumptions And Fragile Areas

These are not necessarily bugs, but future agents should not change them casually.

- `Q_cov` currently estimates selected shock correlations and converts them into covariance entries in `Q`.
- In current two-gap models, `Q_cov_ind[1,3] = true`, and `set_par!` mirrors that covariance to the companion shock entries.
- The same `Q_cov` logic assumes covariance masks refer to Harvey-cycle shock pairs. It is not general for arbitrary trend or non-cycle states.
- Correlation support is hard-coded as `[-0.99, 0.0]` in `mwg_main.jl`. This imposes nonpositive correlations. If positive correlations should be allowed, change `MIN_corr`/`MAX_corr` intentionally and document why.
- `P¹` is initialized block-by-block for stationary Harvey cycles. Cross-block stationary covariance implied by correlated shocks is not currently solved into `P¹`.
- `set_par!` mutates `ParSsm` in place. It must only be called on a safe copy for proposals.

## Notebooks

Primary notebooks:

- `results_two_gap_AR2_6_obs.ipynb`: baseline two-gap result analysis and figures.
- `results_one_gap_models_more_obs.ipynb`: one-gap benchmark analysis.
- `results_hasenzagl_2020.ipynb`: Hasenzagl-style model analysis.
- `results_two_Gap_AR2_6_obs_oos.ipynb`: out-of-sample analysis.

Notebook compatibility rule:

- Old `.jld2` files may contain full `chain_θ_bound` with adaptation columns.
- New `.jld2` files contain only final stored posterior chain columns.
- Notebooks therefore normalize loaded chains to the last `size(distr_α, 3)` draws immediately after loading:

```julia
theta_keep = (size(chain_θ_bound, 2)-size(distr_α, 3)+1):size(chain_θ_bound, 2)
chain_θ_bound = chain_θ_bound[:, theta_keep]
```

Do not remove this unless all old result files are regenerated or deleted.

The OOS notebook expects current output names:

```julia
res_two_gap_AR2_6_obs_oos_chunk0.jld2
res_two_gap_AR2_6_obs_oos_chunk$(i).jld2
```

## Data Flow

1. `user_main.jl` sets `data_path`.
2. `read_data(data_path, model)` returns:
   - `data`
   - `date`
   - `nM`
   - `nQ`
   - `MNEMONIC`
3. `standardise_data!` rescales observables by first-difference volatility.
4. The model wrapper receives standardized data and `σʸ`.
5. Forecast results are later rescaled by `σʸ` where needed.

`data` may contain `missing` values, especially after appending forecast horizons.

## Result File Compatibility

When changing sampler output shape, check:

- `tc_main.jl`
- all `results_*.ipynb`
- any direct indexing of `chain_θ_bound`
- any use of `nDraws`, `burnin`, or `distr_par`

Avoid assuming `chain_θ_bound` has adaptation columns. Prefer:

```julia
n_keep = size(distr_α, 3)
```

or:

```julia
n_keep = length(distr_par)
```

depending on context.

Conditional forecasts should use the loaded result's stored draw count:

```julia
n_draws_final = length(distr_par)
```

not the caller's current `nDraws` settings.

## Common Verification Commands

Load the sampler and all model wrappers:

```powershell
@'
include("code/Metropolis-Within-Gibbs/MetropolisWithinGibbs.jl")
using .MetropolisWithinGibbs
for path in [
    "code/tc_models/tc_hasenzagl_2020.jl",
    "code/tc_models/tc_kuttner_AR2_4_obs.jl",
    "code/tc_models/tc_okun_kuttner_AR2_6_obs.jl",
    "code/tc_models/tc_two_gap_AR2_6_obs.jl",
    "code/tc_models/tc_two_gap_AR2_6_obs_lags.jl",
]
    include(path)
end
println("module and wrappers loaded")
'@ | julia --project=.
```

Check notebooks are valid JSON:

```powershell
@'
import json
from pathlib import Path
for path in Path(".").glob("results*.ipynb"):
    json.loads(path.read_text(encoding="utf-8"))
    print(f"valid {path}")
'@ | python -
```

Search for stale sampler/result assumptions:

```powershell
rg -n "mwg_diagnostics|nDraws\\[3\\]|burnin\\[3\\]|mwg_const\\[3\\]|burnin\\[end\\]\\+1:end|burnin_HZ\\[2\\]\\+1:end|set_par_fast" -S . --glob "!README.md"
```

Run a tiny sampler smoke test:

```powershell
$script = @'
include("code/Metropolis-Within-Gibbs/MetropolisWithinGibbs.jl")
using .MetropolisWithinGibbs
using LinearAlgebra, Random

Random.seed!(11)
y = reshape(sin.(collect(1.0:24.0) ./ 3.0), 1, :)
par = ParSsm(Matrix{Union{Float64, Missing}}(y), zeros(1), [1.0;;], zeros(1, 1),
             zeros(1), [0.7;;], [1.0;;], zeros(1), [1.0;;], zeros(1, 1),
             Float64[], Float64[], 0.0, 0.0, 0.0)
false1 = falses(1)
false11 = falses(1, 1)
par_ind = BoolParSsm(false1, false11, false11, false11, false11, false1,
                     false11, trues(1, 1), false11, falses(0), falses(0))
nDraws = [20; 20]
burnin = [5; 5]
mwg_const = [0.05; 0.10]
res = mwg_main(par, 1, nDraws, burnin, mwg_const, par_ind, [1.0];
               acc_target=0.25, adapt_interval=2)
@assert length(res) == 9
@assert length(res[5]) == 2
@assert size(res[1], 3) == nDraws[end] - burnin[end]
@assert size(res[3], 2) == nDraws[end] - burnin[end]
println("tiny sampler smoke passed")
'@
$script | julia --project=. -e "include_string(Main, read(stdin, String))"
```

## Guidance For Future Agents

- Read `user_main.jl`, `tc_main.jl`, the selected model wrapper, `mwg_main.jl`, `mwg_run.jl`, and `set_par.jl` before changing estimation behavior.
- Do not change parameter ordering casually.
- Do not remove safe `ParSsm` copy behavior.
- Do not use current caller draw settings to interpret old saved results; inspect loaded arrays.
- Keep notebooks compatible with both old full-chain results and new stored-chain-only results.
- Prefer narrow, behavior-preserving patches. Broad style refactors are risky because result notebooks encode many shape assumptions.
