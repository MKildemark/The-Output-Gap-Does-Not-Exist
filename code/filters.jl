
using LinearAlgebra
using DSP
using Statistics
#########################
#  HP Filter
#########################
function HP_filter(signal; λ=1600)
    T = length(signal)
    I_mat = Matrix{Float64}(I, T, T)

    D = zeros(T-2, T)
    for i in 1:(T-2)
        D[i, i] = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    end

    trend_hp = (I_mat + λ * (D'D)) \ signal
    cycle_hp = signal - trend_hp

    return cycle_hp
end



#########################
#  Butterworth Filter
#########################
function butter_filter(signal; fc=0.95, order=1)
    filt = digitalfilter(Lowpass(fc), Butterworth(order))
    # Adjust the gain constant (only affecting the k factor)
    filt_adjusted = 0.89 * filt
    cycle_butter = filtfilt(filt_adjusted, signal)
    return cycle_butter
end


function canova_butterworth_filter(y::AbstractVector{<:Real};
                                order::Int=1,
                                omega::Float64=0.95*pi,
                                G0::Float64=0.4)


    # DSP.jl uses normalized cutoff in (0,1), where 1.0 corresponds to π radians (Nyquist)
    wc_norm = omega / π                 # 0.95 by default
    filt = digitalfilter(Lowpass(wc_norm), Butterworth(order))

    # Zero-phase application (forward–backward) to avoid phase distortion
    y_lp = filtfilt(filt, collect(float.(y)))

    # Scale to achieve the desired (uniform) squared-gain height G0
    amp = sqrt(G0)                      # √0.4 by default
    y_gap = amp .* y_lp

    # Define the residual as the "trend" component
    y_trend = collect(float.(y)) .- y_gap

    return y_gap
end


#########################
#  Polynomial Filter
#########################
function polynomial_filter(y; degree=2)
    # Get the length of the data
    T = length(y)
    
    # Create time variable
    t = 1:T
    
    # Create design matrix for polynomial regression
    X = ones(T)
    for d in 1:degree
        X = [X t.^d]
    end
    
    # Regression to get trend
    β = (X'X)\(X'y)
    trend = X * β
    
    # Cyclical component as residual
    gap = y - trend
    
    return gap
end


########################################################################
#  Hamilton (2018) Filter
########################################################################

function hamilton_filter(y; h::Int = 8, p::Int = 4)
    T      = length(y)
    n_obs  = T - h - p + 1               # effective regression sample


    # ---------- build the forecasting regression ---------------------------
    y_forward =  y[(p + h) : T]                 # y_{t+h}
    X = ones(n_obs, p + 1)                          # constant + p lags
    for j = 1:p
        X[:, j + 1] .=  y[(p - j + 1) : (T - h - j + 1)]
    end

    β = (X'X) \ (X'y_forward)                        # OLS coefficients
    trend_hat = X * β                                # ŷ_{t+h}

    # ---------- cycle: residual at times t + h -----------------------------
    cycle = fill(NaN, T)
    cycle[(p + h) : T] .= y_forward .- trend_hat     # fill the series
    return cycle
end

