#=
This file is part of the replication code for: Hasenzagl, T., Pellegrino, F., Reichlin, L., & Ricco, G. (2020). A Model of the Fed's View on Inflation.
Please cite the paper if you are using any part of the code for academic work (including, but not limited to, conference and peer-reviewed papers).
=#

__precompile__()

module MetropolisWithinGibbs
# ----------------------------------------------------------------------------------------------------------------------
# This module define the functions used to estimate state-space models with a Metropolis-Within-Gibbs algorithm.
#
# The Metropolis step of the Metropolis-Within-Gibbs follows closely the algorithm used in An and Schorfheide (2007).
# The major difference between the two of them is that An and Schorfheide (2007) draw the parameters directly with
# the Metropolis algorithm. Here, the supports of most parameters are constrained in some dimension.
# Therefore, the parameters are transformed, so that their support is unconstrained, and then the Metropolis algorithm
# is executed. The sum of the natural logarithm of the Jacobian is used to evaluate correctly the prior densities.
# DSGE software (e.g. YADA) uses similar modifications to deal with constrained parameters.
# ----------------------------------------------------------------------------------------------------------------------

	using DataFrames, Distributions, Distributed, LinearAlgebra;
	local_path = dirname(@__FILE__);


	# -----------------------------------------------------------------------------------------------------------------
	# Types
	# -----------------------------------------------------------------------------------------------------------------

	mutable struct ParSsm{X <: Float64} # Z_plus and Z_minus should not be included here
		y::Array{Union{X, Missing}, 2}
		d::Union{Array{X, 1}}
		Z::Union{Array{X, 1}, Array{X, 2}}
		R::Union{Array{X, 1}, Array{X, 2}}
		c::Union{Array{X, 1}}
		T::Union{Array{X, 1}, Array{X, 2}}
		Q::Union{Array{X, 1}, Array{X, 2}}
		α¹::Union{Array{X, 1}}
		P¹::Union{Array{X, 1}, Array{X, 2}}
		P̄¹::Union{Array{X, 1}, Array{X, 2}}
		λ::Array{X, 1}
		ρ::Array{X, 1}
		logprior::X
		loglik::X
		logposterior::X
	end

	struct SizeParSsm{X <: Int64}
		d::X
		Z::X
		Z_plus::X
		Z_minus::X
		R::X
		c::X
		T::X
		Q::X
		Q_cov::X
		λ::X
		ρ::X
		θ::X
	end



	struct BoolParSsm{X1 <: BitArray{1}, X2 <: BitArray{2}}
		d::X1
		Z::X2
		Z_plus::X2
		Z_minus::X2
		R::X2
		c::X1
		T::X2
		Q::X2
		Q_cov::X2
		λ::X1
		ρ::X1
	end

	struct PriorOpt{X <: Float64} # the logpdf for λ and ρ are constants
		N::Distributions.Normal{X}
		N_plus::Distributions.Truncated{Normal{X}, Continuous, X}
		N_minus::Distributions.Truncated{Normal{X}, Continuous, X}
		IG::Distributions.InverseGamma{X}
		λ::X
		ρ::X
		corr::X
	end

	import Base.copy
	Base.copy(x::ParSsm) = ParSsm(copy(x.y), copy(x.d), copy(x.Z), copy(x.R), copy(x.c),
	                              copy(x.T), copy(x.Q), copy(x.α¹), copy(x.P¹), copy(x.P̄¹),
	                              copy(x.λ), copy(x.ρ), x.logprior, x.loglik, x.logposterior)
	Base.copy(x::SizeParSsm) = SizeParSsm(x.d, x.Z, x.Z_plus, x.Z_minus, x.R, x.c, x.T, x.Q,
	                                      x.Q_cov, x.λ, x.ρ, x.θ)
	Base.copy(x::BoolParSsm) = BoolParSsm(copy(x.d), copy(x.Z), copy(x.Z_plus), copy(x.Z_minus),
	                                      copy(x.R), copy(x.c), copy(x.T), copy(x.Q),
	                                      copy(x.Q_cov), copy(x.λ), copy(x.ρ))
	Base.copy(x::PriorOpt) = PriorOpt(x.N, x.N_plus, x.N_minus, x.IG, x.λ, x.ρ, x.corr)

	# -----------------------------------------------------------------------------------------------------------------
	# Subroutines
	# -----------------------------------------------------------------------------------------------------------------

	# Estimation
	include("$local_path/subroutines/estimation/kalman_diffuse!.jl");
	include("$local_path/subroutines/estimation/mwg_main.jl");
	include("$local_path/subroutines/estimation/mwg_run.jl");

	# Julia's subroutines
	include("$local_path/subroutines/extra/ex_blkdiag.jl");
	include("$local_path/subroutines/extra/ex_inv.jl");
	include("$local_path/subroutines/extra/ex_ismember.jl");

	# Get
	include("$local_path/subroutines/get/get_logjacobian.jl");
	include("$local_path/subroutines/get/get_par_bound.jl");
	include("$local_path/subroutines/get/get_par_unb.jl");
	include("$local_path/subroutines/get/get_progress.jl");
	include("$local_path/subroutines/get/get_random_disturbance.jl");
	include("$local_path/subroutines/get/get_mwg_jump.jl");

	# Set
	include("$local_path/subroutines/set/set_par.jl");


	# -----------------------------------------------------------------------------------------------------------------
	# Export functions
	# -----------------------------------------------------------------------------------------------------------------

	export kalman_diffuse!, mwg_main, ex_blkdiag, ex_inv, ex_ismember, set_par!, ParSsm, SizeParSsm, BoolParSsm, PriorOpt;
end
