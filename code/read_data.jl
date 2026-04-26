#=
This file is part of the replication code for: Hasenzagl, T., Pellegrino, F., Reichlin, L., & Ricco, G. (2020). A Model of the Fed's View on Inflation.
Please cite the paper if you are using any part of the code for academic work (including, but not limited to, conference and peer-reviewed papers).
=#

include("./quarterly2monthly.jl");

function read_data(data_path::AbstractString, model::AbstractString)
    # Defaults
    date = Date[]
    data = Array{Union{Missing, Float64}}(undef, 0, 0)
    nM   = 0
    nQ   = 0
    MNEMONIC = String[]


    # column positions in the "quarterly" sheet
    dates          = 2
    gdp            = 3
    employment     = 4
    unemployment   = 5
    oil            = 6                # not used 
    inflation      = 7               # π_t
    core_inflation = 8               # not used
    UoM            = 9               # E_t^{UoM} π_{t+4}
    SPF            = 10              # E_t^{SPF} π_{t+4}
    ten_year_inflation = 11          # E_t^{LT} π_{t+40}
    one_year_inflation = 12          # E_t^{ST} π_{t+4}
    thirty_year_inflation = 13      # E_t^{VLT} π_{t+120}


    fQ = DataFrame(XLSX.readtable(data_path, "quarterly"))
    # rows
    rows = 18:size(fQ, 1)
    # Dates
    date = fQ[rows, dates]|> Array{Date,1}
    

 
    if model == "two_gap_AR2_6_obs" || model == "two_gap_AR2_6_obs_lags" || model == "okun_kuttner_AR2_6_obs"
        # Pull the level series (as Float64/ Missing vectors)
        y   = Vector{Union{Missing, Float64}}(fQ[rows, gdp])
        e   = Vector{Union{Missing, Float64}}(fQ[rows, employment])
        u   = Vector{Union{Missing, Float64}}(fQ[rows, unemployment])
        π   = Vector{Union{Missing, Float64}}(fQ[rows, inflation])
        core_π = Vector{Union{Missing, Float64}}(fQ[rows, core_inflation])
        uom = Vector{Union{Missing, Float64}}(fQ[rows, UoM])
        spf = Vector{Union{Missing, Float64}}(fQ[rows, SPF])

        y = log.(y)*100
        e = log.(e)*100

        # Final matrix: 7 columns 
        data_quarterly = hcat(y, e, u, π, uom, spf)
        # data_quarterly = hcat(y, e, u, π, spf)

        info_data = DataFrame(XLSX.readtable(data_path, "transf"))
        MNEMONIC  = info_data[1:end, 2] |> Array{String,1};
        MNEMONIC = vcat(MNEMONIC[1:3], MNEMONIC[6:8]);

    elseif model == "kuttner_AR2_4_obs"  
        # Pull the level series (as Float64/ Missing vectors)
        y   = Vector{Union{Missing, Float64}}(fQ[rows, gdp])
        y = log.(y)*100
        π   = Vector{Union{Missing, Float64}}(fQ[rows, inflation])
        uom = Vector{Union{Missing, Float64}}(fQ[rows, UoM])
        spf = Vector{Union{Missing, Float64}}(fQ[rows, SPF])

        # Final matrix: 3 columns 
        data_quarterly = hcat(y, π, uom, spf)

        info_data = DataFrame(XLSX.readtable(data_path, "transf"))
        MNEMONIC  = info_data[1:end, 2] |> Array{String,1};
        MNEMONIC = vcat(MNEMONIC[1:2], MNEMONIC[6]);

    elseif model == "hasenzagl_2020"
        y      = Vector{Union{Missing, Float64}}(fQ[rows, gdp])
        e      = Vector{Union{Missing, Float64}}(fQ[rows, employment])
        u      = Vector{Union{Missing, Float64}}(fQ[rows, unemployment])
        oil_p  = Vector{Union{Missing, Float64}}(fQ[rows, oil])
        π      = Vector{Union{Missing, Float64}}(fQ[rows, inflation])
        core_π = Vector{Union{Missing, Float64}}(fQ[rows, core_inflation])
        uom    = Vector{Union{Missing, Float64}}(fQ[rows, UoM])
        spf    = Vector{Union{Missing, Float64}}(fQ[rows, SPF])

        y = log.(y) * 100
        e = log.(e) * 100

        data_quarterly = hcat(y, e, u, oil_p, π, core_π, uom, spf)

        info_data = DataFrame(XLSX.readtable(data_path, "transf"))
        MNEMONIC  = info_data[1:end, 2] |> Array{String,1}

    else
        error("Model not recognized")

       
    end
    # Convert to the expected element type
    data = Array{Union{Missing, Float64}}(data_quarterly)
    nQ = size(data, 2)


    return data, date, nM, nQ, MNEMONIC
end
