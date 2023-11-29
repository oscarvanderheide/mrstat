function steihaug(H, g, Δ, P, maxit, tol, z0)

    # This implementation is really just a copy-paste of Algorithm 7.2
    # of Nocedal & Wright

    # H  = Hessian multiplication function
    # r  = Current residual
    # d  = Current B-conjugate search direction
    # z  = Linear combination of previous B-conjugate directions

    println("    Steihaug CG:")
    ϵ = eps();
    η = min( tol, norm(g) / length(g) ); # Should give quadratic convergence near solution
    # η = min( 0.5, sqrt(norm(g)) ); # Should give superlinear convergence near solution

    tol = η * norm(g);
    # println("        Current error tolerance for Steihaug is $(tol)");
    # println("        Current trust radius is $(Δ)", );
    # Initialize things for CG algorithm
    z = zeros(length(g));
    r = g

    Y = P(r);
    d = -Y;

    # initialize empty array to store Steihaug steps
    steps = eltype(g)[]
    sizehint!(steps, maxit*length(d))

    if norm(r) < tol
        println("        Nothing to gain, residual is already small enough from the start")
        append!(steps,z);
    end

    iter = 1;

    while iter <= maxit

        print(".")

        # println("        Iteration $(iter) of inner loop, current CG-residual: $(norm(r))")
        Hd = H(d);
        dHd = d' * Hd;
        #  realResidual = 0.5 * p' * B(p) + g' * p    # This thing
        #  should be monotonically decreasing (and it does)

        # if dHd < ϵ
        if dHd < ϵ
            println("        Direction of negative curvature encountered: should not occur because of Gauss-Newton method?")
            τ = distanceToTrustRegion(z, d, Δ);
            append!(steps,z + τ * d);
            break
        end

        α = (r' * Y) / dHd;
        z_new = z + α * d;

        if norm(z_new) > Δ
            println("        Fell out of trust radius after iteration $(iter)" )
            τ = distanceToTrustRegion(z, d, Δ);
            append!(steps,z + τ * d);
            break
        end

        r_new = r + α * Hd;
        norm_r_new = norm(r_new)

        if norm_r_new < tol
            println("        Steihaug-CG converged with CG-residual = $(norm_r_new) after iteration $(iter)");
            append!(steps,z_new);
            break
        end

        Y_new = P(r_new);
        β    = (Y_new' * r_new)  / (Y' * r);
        d_new   = -Y_new + β * d;

        # Prepare for next iteration
        r = r_new;
        d = d_new;
        z = z_new;
        Y = Y_new;

        if iter == maxit
            println("        Steihaug-CG failed to converge, CG-residual = $(norm_r_new)")
            append!(steps,z_new);
            break
        else
            iter = iter + 1;
            append!(steps,z_new)
        end
    end

    steps = reshape(steps,length(g),:)

    return steps

end