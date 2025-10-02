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


    fQ = DataFrame(XLSX.readtable(data_path, "quarterly"))
    # rows
    rows = 18:size(fQ, 1)
    # Dates
    date = fQ[rows, dates]|> Array{Date,1}
    

    if model == "full" || model == "full_no_lags"
        # Pull the level series (as Float64/ Missing vectors)
        y   = Vector{Union{Missing, Float64}}(fQ[rows, gdp])
        e   = Vector{Union{Missing, Float64}}(fQ[rows, employment])
        u   = Vector{Union{Missing, Float64}}(fQ[rows, unemployment])
        π   = Vector{Union{Missing, Float64}}(fQ[rows, inflation])
        uom = Vector{Union{Missing, Float64}}(fQ[rows, UoM])
        spf = Vector{Union{Missing, Float64}}(fQ[rows, SPF])

        y = log.(y)*100
        e = log.(e)*100

        # Build the difference series elementwise
        π_minus_uom = π .- uom       # π_t − E_t^{UoM} π_{t+4}
        π_minus_spf = π .- spf       # π_t − E_t^{SPF} π_{t+4}

        # Final matrix: 5 columns 
        data_quarterly = hcat(y, e, u, π_minus_uom, π_minus_spf)

        info_data = DataFrame(XLSX.readtable(data_path, "transf"))
        MNEMONIC  = info_data[1:end, 2] |> Array{String,1};
        # drop 4th element (OIL), 5th element (CPI) and 7th element (Core CPI)
        MNEMONIC = vcat(MNEMONIC[1:3], MNEMONIC[6:7]);
        
    elseif model == "full_inf" || model == "full_inf_no_lags"
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

        info_data = DataFrame(XLSX.readtable(data_path, "transf"))
        MNEMONIC  = info_data[1:end, 2] |> Array{String,1};
        MNEMONIC = vcat(MNEMONIC[1:3], MNEMONIC[6:8]);
    elseif model == "inflation"
        # Pull the level series (as Float64/ Missing vectors)
        y   = Vector{Union{Missing, Float64}}(fQ[rows, gdp])
        y = log.(y)*100
        π   = Vector{Union{Missing, Float64}}(fQ[rows, inflation])

        # Final matrix: 2 columns 
        data_quarterly = hcat(y, π)

        info_data = DataFrame(XLSX.readtable(data_path, "transf"))
        MNEMONIC  = info_data[1:end, 2] |> Array{String,1};
        MNEMONIC = vcat(MNEMONIC[1:2], MNEMONIC[6]);
    elseif model == "employment"
        # Pull the level series (as Float64/ Missing vectors)
        y   = Vector{Union{Missing, Float64}}(fQ[rows, employment])
        y = log.(y)
        u   = Vector{Union{Missing, Float64}}(fQ[rows, unemployment])

        # Final matrix: 2 columns
        data_quarterly = hcat(y, u)

        info_data = DataFrame(XLSX.readtable(data_path, "transf"))
        MNEMONIC  = info_data[1:end, 2] |> Array{String,1};
        MNEMONIC = vcat(MNEMONIC[1:2], MNEMONIC[4]);
    elseif model == "employment+inflation"
        # Pull the level series (as Float64/ Missing vectors)
        y   = Vector{Union{Missing, Float64}}(fQ[rows, employment])
        y = log.(y)*100
        u   = Vector{Union{Missing, Float64}}(fQ[rows, unemployment])
        π   = Vector{Union{Missing, Float64}}(fQ[rows, inflation])

        # Final matrix: 3 columns 
        data_quarterly = hcat(y, u, π)

        info_data = DataFrame(XLSX.readtable(data_path, "transf"))
        MNEMONIC  = info_data[1:end, 2] |> Array{String,1};
        MNEMONIC = vcat(MNEMONIC[1:2], MNEMONIC[4], MNEMONIC[6]);
    
    elseif model == "flex"
        # Pull the level series (as Float64/ Missing vectors)
        y   = Vector{Union{Missing, Float64}}(fQ[rows, gdp])
        y = log.(y)*100
        π   = Vector{Union{Missing, Float64}}(fQ[rows, inflation])
        uom = Vector{Union{Missing, Float64}}(fQ[rows, UoM])
        spf = Vector{Union{Missing, Float64}}(fQ[rows, SPF])

        # Build the difference series elementwise
        π_minus_uom = π .- uom       # π_t − E_t^{UoM} π_{t+4}
        π_minus_spf = π .- spf       # π_t − E_t^{SPF} π_{t+4}

        # Final matrix: 5 columns in the requested order
        data_quarterly = hcat(y, π_minus_uom, π_minus_spf)

        info_data = DataFrame(XLSX.readtable(data_path, "transf"))
        MNEMONIC  = info_data[1:end, 2] |> Array{String,1};
        MNEMONIC = vcat(MNEMONIC[1:3], MNEMONIC[6:7]);

    elseif model == "efficient"
        # Pull the level series (as Float64/ Missing vectors)
        y   = Vector{Union{Missing, Float64}}(fQ[rows, gdp])
        y = log.(y)*100
        e   = Vector{Union{Missing, Float64}}(fQ[rows, employment])
        u   = Vector{Union{Missing, Float64}}(fQ[rows, unemployment])
       

        # Final matrix: 5 columns in the requested order
        data_quarterly = hcat(y, e, u)

        info_data = DataFrame(XLSX.readtable(data_path, "transf"))
        MNEMONIC  = info_data[1:end, 2] |> Array{String,1};
        MNEMONIC = vcat(MNEMONIC[1:3], MNEMONIC[6:7]);

    else
        error("Model not recognized. Choose either 'full' or 'simple'.")

       
    end
    # Convert to the expected element type
    data = Array{Union{Missing, Float64}}(data_quarterly)
    nQ = size(data, 2)


    return data, date, nM, nQ, MNEMONIC
end
