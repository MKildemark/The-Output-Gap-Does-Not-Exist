using LinearAlgebra
# this needs to be changes to match the model as implemented in set_par
function build_Z_for_draw(θ::Vector{Float64},
                          par_ind,
                          par_size)

    # dimensions from the indicator matrices
    n, n_states = size(par_ind.Z)   # n = #observables, n_states = #states

    Z  = zeros(n, n_states)
    T  = zeros(n_states, n_states)
    λv = zeros(n_states)
    ρv = zeros(n_states)

    # we only need ϕ[1] for NKPC; store it
    ϕ = zeros(n_states)   # or just scalar if you know it’s one parameter

    iend = 0

    # --- R block (skip filling; we only need its size to move iend correctly)
    if par_size.R > 0
        iend += par_size.R
    end

    # --- d block (skip)
    if par_size.d > 0
        iend += par_size.d
    end

    # --- ϕ block (we DO need it for β in NKPC)
    if par_size.ϕ > 0
        ϕ[par_ind.ϕ .== true] .= θ[iend+1 : iend+par_size.ϕ]
        iend += par_size.ϕ
    end

    # --------------------------- Z, Z_plus, Z_minus ---------------------------

    # Z free entries
    if par_size.Z > 0
        Z[par_ind.Z .== true] .= θ[iend+1 : iend+par_size.Z]
        iend += par_size.Z
    end

    # Z_plus (sign-restricted)
    if par_size.Z_plus > 0
        Z[par_ind.Z_plus .== true] .= θ[iend+1 : iend+par_size.Z_plus]
        iend += par_size.Z_plus
    end

    # Z_minus (negatively restricted)
    if par_size.Z_minus > 0
        Z[par_ind.Z_minus .== true] .= θ[iend+1 : iend+par_size.Z_minus]
        iend += par_size.Z_minus
    end



    # --------------------------- skip Q, Q_cov, H, c ---------------------------

    if par_size.Q > 0
        iend += par_size.Q
    end

    if par_size.Q_cov > 0
        iend += par_size.Q_cov
    end

    if par_size.H > 0
        iend += par_size.H
    end

    if par_size.c > 0
        iend += par_size.c
    end

    # --------------------------- T, λ, ρ as in set_par! ---------------------------

    # T free
    if par_size.T > 0
        T[par_ind.T .== true] .= θ[iend+1 : iend+par_size.T]
        iend += par_size.T
    end

    # trig part: λ and ρ and update T blocks
    if par_size.λ > 0 || par_size.ρ > 0
        # λ
        if par_size.λ > 0
            λv[par_ind.λ .== true] .= θ[iend+1 : iend+par_size.λ]
        end
        # ρ
        if par_size.ρ > 0
            ρv[par_ind.ρ .== true] .= θ[iend+par_size.λ+1 : iend+par_size.λ+par_size.ρ]
        end

        # IMPORTANT: this is the same logic you have in set_par!
        bool_trig = ((par_ind.λ .== true) .+ (par_ind.ρ .== true)) .> 0
        find_trig = findall(bool_trig)

        for i = 1:sum(bool_trig)
            j = find_trig[i]
            T[j:j+1, j:j+1] =
                ρv[j] * [  cos(λv[j])  sin(λv[j]);
                         -sin(λv[j])  cos(λv[j]) ]
        end

        # ---------------------- RE NKPC override of π row ----------------------

        # Same condition as in your set_par!:
        # size(Z,1) == 4 or 5 && par_size.λ > 0 && par_size.Z_plus == 1
        if (size(Z, 1) == 4 || size(Z, 1) == 5) && par_size.λ > 0 && par_size.Z_plus == 1

            # Discount factor β from ϕ[1] (or 0.99 fallback)
            β = ϕ[1]

            # κ is the *first* Z_plus parameter in θ (same indexing as in set_par!)
            idx_κ = par_size.R + par_size.d + par_size.ϕ + par_size.Z + 1
            κ     = θ[idx_κ]

            # identify which state indices are the two Harvey cycles
            # (this must match your state ordering in tc_two_gap_AR2_rational.jl).
            #
            # In your set_par! we used:
            # j_gap  = 1    # Ψᵉ cycle has states 1–2
            # j_cost = 3    # Ψ^π cycle has states 3–4
            j_gap  = 1
            j_cost = 3

            Φ_gap  = T[j_gap:j_gap+1,   j_gap:j_gap+1]
            Φ_cost = T[j_cost:j_cost+1, j_cost:j_cost+1]

            I2 = Matrix{Float64}(I, 2, 2)

            # A_c'  = κ e1' (I - β Φ_gap)^(-1)
            # A_u'  =     e1' (I - β Φ_cost)^(-1)
            A_gap  = κ * [1.0 0.0] * inv(I2 .- β * Φ_gap)    # 1×2
            A_cost =      [1.0 0.0] * inv(I2 .- β * Φ_cost)  # 1×2

            # π is observable 4: row_π = 4
            row_π = 4

            # overwrite π row loadings on the two cycles (exactly as in set_par!)
            Z[row_π, j_gap:j_gap+1]   .= A_gap[1, 1:2]
            Z[row_π, j_cost:j_cost+1] .= A_cost[1, 1:2]
            Z[1,1] = 1.0
            Z[4,5] = 1.0

            # If long-run inflation expectations are observed, use RE-implied E_t[π_{t+40}]
            if size(Z, 1) == 5
                row_LT = 5
                H_LT   = 40

                A_gap_LT  = A_gap  * (Φ_gap^H_LT)
                A_cost_LT = A_cost * (Φ_cost^H_LT)

                Z[row_LT, j_gap:j_gap+1]   .= A_gap_LT[1, 1:2]
                Z[row_LT, j_cost:j_cost+1] .= A_cost_LT[1, 1:2]
                Z[row_LT, 5]                = 1.0
            end
        end
    end

    return Z
end
