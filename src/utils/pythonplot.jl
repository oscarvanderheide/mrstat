
lipari = QMRIColors.relaxationColorMap("T1")
navia = QMRIColors.relaxationColorMap("T2")


function plot_T₁T₂ρ(x::AbstractArray{<:AbstractTissueProperties}, Nx, Ny, figtitle="")

    # translating the colormap to a format digestible by PythonPlot
    cmap_lipari = PythonPlot.ColorMap("lipari", lipari, length(lipari), 1.0)
    cmap_navia = PythonPlot.ColorMap("lipari", navia, length(navia), 1.0)

    q = StructArray(reshape(x, Nx, Ny))

    figure()

    subplot(131)
    imshow(q.T₁, clim=(0.0, 2.5), cmap=cmap_lipari)
    colorbar()
    xlabel("T₁ [s]")
    subplot(132)
    imshow(q.T₂, clim=(0.0, 0.35), cmap=cmap_navia)
    colorbar()
    xlabel("T₂ [s]")
    subplot(133)
    imshow(abs.(complex.(q.ρˣ, q.ρʸ)), clim=(0.0, 2.0), cmap="gray")
    colorbar()
    xlabel("ρ [a.u.]")

    suptitle(figtitle)

end
