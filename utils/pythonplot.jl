# make colormaps
loLevT₁ = 0.0; upLevT₁ = 2.5;
loLevT₂ = 0.0; upLevT₂ = 0.35;
img = 0.1*ones(10,10)
_, rgb_vec_T₁ = relaxationColorMap("T1", img, loLevT₁, upLevT₁)
_, rgb_vec_T₂ = relaxationColorMap("T2", img, loLevT₂, upLevT₂)
lipari = PythonPlot.ColorMap("lipari", rgb_vec_T₁, length(rgb_vec_T₁), 1.0)
navia  = PythonPlot.ColorMap("navia",  rgb_vec_T₂, length(rgb_vec_T₂), 1.0)

function plot_T₁T₂ρ(x::AbstractArray{<:AbstractTissueParameters}, Nx, Ny, figtitle="")

    q = StructArray(reshape(x,Nx,Ny))

    figure()

    subplot(131)
        imshow(q.T₁, clim=(0.0,2.5), cmap=lipari)
        colorbar()
        xlabel("T₁ [s]")
    subplot(132)
        imshow(q.T₂, clim=(0.0,0.35), cmap=navia)
        colorbar()
        xlabel("T₂ [s]")
    subplot(133)
        imshow(abs.(complex.(q.ρˣ, q.ρʸ)), clim=(0.0,2.0), cmap="gray")
        colorbar()
        xlabel("ρ [a.u.]")

    suptitle(figtitle)

end
