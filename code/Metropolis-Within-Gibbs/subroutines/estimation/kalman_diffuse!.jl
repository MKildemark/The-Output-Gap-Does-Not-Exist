

function kalman_diffuse!(par::ParSsm, do_loglik::Int64=0, do_smoother::Int64=0, do_sim_smoother::Int64=0, F_tol::Float64=1e-8)

    # ----------------------------- Initialisation -----------------------------
    k    = size(par.T)[1]
    n, m = size(par.y)

    # Kalman filter output
    ν   = zeros(m, n)
    K   = zeros(k, n, m)
    K̄   = zeros(k, n, m)
    F   = zeros(m, n)
    F̄   = zeros(m, n)
    α̂ᶠ  = zeros(k, m)
    Pᶠ  = zeros(k, k, m)
    P̄ᶠ  = zeros(k, k, m)

    # Kalman smoother output
    α̂ˢ = zeros(k, m)
    Pˢ  = zeros(k, k, m)
    P̄ˢ = zeros(k, k, m)

    # Loglikelihood
    if do_loglik == 1
        par.loglik = 0.0
    end
    if do_sim_smoother == 1
        do_smoother = 1
    end

    # Initial variance for diffuse states
    par.P¹[par.P̄¹ .== 1] .= 0.0

    # ----------------------- Simulation smoother draws ------------------------
    if do_sim_smoother == 1
        y⁺ = zeros(n, m)
        α⁺ = zeros(k, m)

        # Durbin-Koopman (demeaned model) with Jarociński mod;  u_t = H * ε_t
        for t = 1:m
            if t == 1
                α⁺[:, 1] = get_random_disturbance(k, par.P¹)
            else
                ε = get_random_disturbance(k, par.Q)              ## H-update
                α⁺[:, t] = par.T * α⁺[:, t-1] + par.H * ε         ## H-update
            end
            y⁺[:, t] = par.Z * α⁺[:, t] + get_random_disturbance(n, par.R)
        end
        yᵃ = par.y - y⁺
    else
        yᵃ = par.y
    end

    # --------- Handle correlated measurement errors (Schur pre-whitening) -----
    R_ind = par.R .!= 0
    R_ind[Matrix(I, n, n) .== 1] .= false

    if sum(sum(R_ind)) > 0
        H_R, S, _ = schur(copy(par.R))
        yᵃ = S' * yᵃ
        d̃  = S' * copy(par.d)
        Z̃  = S' * copy(par.Z)
        R̃  = H_R
    else
        d̃ = par.d
        Z̃ = par.Z
        R̃ = par.R
    end

    # --------------------------- Kalman Filter --------------------------------
    α̂ᶠᵢ = par.α¹
    Pᶠᵢ  = par.P¹
    P̄ᶠᵢ = par.P̄¹

    α̂ᶠ[:, 1]   = α̂ᶠᵢ
    Pᶠ[:, :, 1] = Pᶠᵢ
    P̄ᶠ[:, :, 1]= P̄ᶠᵢ

    for t = 1:m
        α̂ᶠᵢ = α̂ᶠ[:, t]
        Pᶠᵢ  = Pᶠ[:, :, t]
        P̄ᶠᵢ = P̄ᶠ[:, :, t]

        # Missing data mask
        no_na     = .!ismissing.(yᵃ[:, t])
        ind_no_na = findall(no_na)
        yᵃᵗ       = convert(Array{Float64,1}, yᵃ[no_na, t])
        d̃ᵗ        = d̃[no_na, :]
        Z̃ᵗ        = Z̃[no_na, :]
        R̃ᵗ        = R̃[no_na, no_na]

        # Sequential (univariate) measurement updates
        for i = 1:sum(no_na)
            Fᵗᵢ  = (Z̃ᵗ[i, :]' * Pᶠᵢ * Z̃ᵗ[i, :] + R̃ᵗ[i, i])[1]
            F̄ᵗᵢ  = (Z̃ᵗ[i, :]' * P̄ᶠᵢ * Z̃ᵗ[i, :])[1]

            if F̄ᵗᵢ > F_tol || Fᵗᵢ > F_tol
                Kᵗᵢ  = Pᶠᵢ * Z̃ᵗ[i, :]
                K̄ᵗᵢ  = P̄ᶠᵢ * Z̃ᵗ[i, :]
                νᵗᵢ  = (yᵃᵗ[i, :] .- d̃ᵗ[i, :] .- Z̃ᵗ[i, :]' * α̂ᶠᵢ)[1]

                if F̄ᵗᵢ > F_tol
                    if t < m
                        α̂ᶠᵢ = α̂ᶠᵢ + K̄ᵗᵢ / F̄ᵗᵢ * νᵗᵢ
                        Pᶠᵢ  = Pᶠᵢ + K̄ᵗᵢ * K̄ᵗᵢ' * Fᵗᵢ / (F̄ᵗᵢ^2) - (Kᵗᵢ * K̄ᵗᵢ' + K̄ᵗᵢ * Kᵗᵢ') / F̄ᵗᵢ
                        P̄ᶠᵢ = P̄ᶠᵢ - K̄ᵗᵢ / F̄ᵗᵢ * K̄ᵗᵢ'
                    end
                    if do_loglik == 1
                        par.loglik -= 0.5 * (log(2π) + log(F̄ᵗᵢ))
                    end
                else
                    if t < m
                        α̂ᶠᵢ = α̂ᶠᵢ + Kᵗᵢ / Fᵗᵢ * νᵗᵢ
                        Pᶠᵢ  = Pᶠᵢ - Kᵗᵢ / Fᵗᵢ * Kᵗᵢ'
                    end
                    if do_loglik == 1
                        par.loglik -= 0.5 * (log(2π) + log(Fᵗᵢ) + (νᵗᵢ^2) / Fᵗᵢ)
                    end
                end

                # Store per-equation outputs for smoother
                if do_smoother == 1
                    ν[t, ind_no_na[i]]     = νᵗᵢ
                    K[:, ind_no_na[i], t]  = Kᵗᵢ
                    K̄[:, ind_no_na[i], t] = K̄ᵗᵢ
                    F[t, ind_no_na[i]]     = Fᵗᵢ
                    F̄[t, ind_no_na[i]]     = F̄ᵗᵢ
                end
            end

            # One-step-ahead prediction
            if t < m
                α̂ᶠ[:, t+1]    = par.c + par.T * α̂ᶠᵢ
                Pᶠ[:, :, t+1]  = par.T * Pᶠᵢ * par.T' + par.H * par.Q * par.H'  ## H-update
                P̄ᶠ[:, :, t+1] = par.T * P̄ᶠᵢ * par.T'
            end
        end

        # If all obs missing at t, still propagate
        if sum(no_na) == 0 && t < m
            α̂ᶠ[:, t+1]    = par.c + par.T * α̂ᶠᵢ
            Pᶠ[:, :, t+1]  = par.T * Pᶠᵢ * par.T' + par.H * par.Q * par.H'      ## H-update
            P̄ᶠ[:, :, t+1] = par.T * P̄ᶠᵢ * par.T'
        end
    end

    # ----------------------------- Kalman Smoother ----------------------------
    if do_smoother == 1
        rᵢ = zeros(2k)
        Nᵢ = zeros(2k, 2k)
        TT = kron(Matrix(I, 2, 2), par.T)

        for t = m:-1:1
            if t < m
                rᵢ = TT' * rᵢ
                Nᵢ = TT' * Nᵢ * TT
            end

            no_na     = .!ismissing.(yᵃ[:, t])
            ind_no_na = findall(no_na)
            Z̃ᵗ        = Z̃[no_na, :]

            for i = sum(no_na):-1:1
                Fᵗᵢ = F[t, ind_no_na[i]]
                F̄ᵗᵢ = F̄[t, ind_no_na[i]]

                if F̄ᵗᵢ > F_tol || Fᵗᵢ > F_tol
                    νᵗᵢ = ν[t, ind_no_na[i]]
                    Kᵗᵢ = K[:, ind_no_na[i], t]
                    K̄ᵗᵢ = K̄[:, ind_no_na[i], t]

                    if F̄ᵗᵢ > F_tol
                        Lᵗᵢ  = (K̄ᵗᵢ * Fᵗᵢ / F̄ᵗᵢ - Kᵗᵢ) * Z̃ᵗ[i, :]' / F̄ᵗᵢ
                        L̄ᵗᵢ  = Matrix(I, k, k) - K̄ᵗᵢ * Z̃ᵗ[i, :]' / F̄ᵗᵢ
                        Mᵗᵢ  = [L̄ᵗᵢ Lᵗᵢ; zeros(size(L̄ᵗᵢ)) L̄ᵗᵢ]
                        temp¹ = Z̃ᵗ[i, :] / F̄ᵗᵢ
                        temp² = temp¹ * Z̃ᵗ[i, :]'
                        rᵢ    = [zeros(k); temp¹ * νᵗᵢ] + Mᵗᵢ' * rᵢ
                        Nᵢ    = [zeros(k, k) temp²; temp² temp² * Fᵗᵢ / F̄ᵗᵢ] + Mᵗᵢ' * Nᵢ * Mᵗᵢ
                    else
                        Lᵗᵢ  = Matrix(I, k, k) - Kᵗᵢ * Z̃ᵗ[i, :]' / Fᵗᵢ
                        Mᵗᵢ  = [Lᵗᵢ zeros(size(Lᵗᵢ)); zeros(size(Lᵗᵢ)) Lᵗᵢ]
                        temp¹ = Z̃ᵗ[i, :] / Fᵗᵢ
                        rᵢ    = [temp¹ * νᵗᵢ; zeros(k)] + Mᵗᵢ' * rᵢ
                        Nᵢ    = [temp¹ * Z̃ᵗ[i, :]' zeros(k, k); zeros(k, 2k)] + Mᵗᵢ' * Nᵢ * Mᵗᵢ
                    end
                end
            end

            P̃ᶠ          = [Pᶠ[:, :, t] P̄ᶠ[:, :, t]]
            α̂ˢ[:, t]    = α̂ᶠ[:, t] + P̃ᶠ * rᵢ
            Pˢ[:, :, t]  = Pᶠ[:, :, t] - P̃ᶠ * Nᵢ * P̃ᶠ'
        end
    end

    # ------------------------------- Return -----------------------------------
    if do_sim_smoother == 1
        α = α̂ˢ + α⁺
        P = Pˢ
    elseif do_smoother == 1
        α = α̂ˢ
        P = Pˢ
    else
        α = α̂ᶠ
        P = Pᶠ
    end

    return α, P
end
