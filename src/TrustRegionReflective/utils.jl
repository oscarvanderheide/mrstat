function distanceToFeasibleRegion(x, s, LB, UB)
    # The function computes a positive scalar stepsize, such that x + stepsize * s is on the bound.
    # boundaryHit = 0 (bound not hit), -1 (lower bound hit), or 1 (upper bound hit)

    non_zero = s .!= 0;
    steps = Inf * ones(length(x));
    steps[non_zero] = max.( (LB[non_zero] - x[non_zero]) ./ s[non_zero], (UB[non_zero] - x[non_zero]) ./ s[non_zero]);
    stepsize = minimum(steps);
    # boundary_hit = (steps == stepsize) .* sign(s);
    boundary_hit = (steps .== stepsize);

    return stepsize, boundary_hit
end

function distanceToTrustRegion(x, p, trustRadius)

    # The function tau --> m(x + tau * p) is decreasing.
    # We need to find tau such that norm( x + tau * p) = trustRadius, or,
    # equivalently, norm( x + tau * p)^2 = trustRadius^2
    # We can simply use the abc-formula to do this :)
    a = norm(p)^2;
    b = 2 * x' * p;
    c = norm(x)^2 - trustRadius^2;

    τ = (-b + sqrt( b^2 - 4*a*c) ) / (2*a);
    return τ
end

function adjustTrustRadius( ratio, step, Δ, min_ratio)

    norm_step = norm(step)
    println("        Norm of step: $(norm_step), Trust Radius: $(Δ)")

    if ratio < min_ratio # The trust region is too large. Reduce the radius.
        println("    Trust Radius too large")
        Δ = (1/4) * Δ;
    elseif (ratio > 1/2) && (norm( step ) > ( 0.95 * Δ) ) # The trust region seems to be too small
        println("    Trust Radius too small")
        Δ = 2 * Δ;
    else # The trust region seems to be fine
        println("    Trust Radius just fine")
    end

    return Δ
end

function computeDistanceToBoundaries(x, g, LB, UB)

    # First-Order optimality condition in the case of box constraints is:
    #
    # D^2 .* g(x) = v(x) .* g(x) = 0,
    #
    # with g the gradient of f and v as computed below.
    #
    # All of this is just a complicated way of saying that the gradient should be
    # positive at a lower bound and negative at an upper bound. Otherwise
    # we could decrease the objective function by moving away from the bounds and we would not be at an optimum.

    # v contains distances to boundaries to which the negative gradient points

        v = ones(eltype(g), length(g));
        # Distances to upper bounds
        v[ (g .< 0) .& (UB .<  Inf) ] = UB[ (g .< 0) .& (UB .< Inf)  ] -  x[ (g .< 0) .& (UB .< Inf)  ];
        # Distances to lower bounds
        v[ (g .> 0) .& (LB .> -Inf) ] =  x[ (g .> 0) .& (LB .> -Inf) ] - LB[ (g .> 0) .& (LB .> -Inf) ];
        # Distance remains 1 for parameters that have no bounds (no scaling needed)

        if any( v .< 0 )
            println("    Somehow x is not within the bounds")
            return
        end


    # We also compute the partial derivatives of v
        dv = zeros(eltype(v), length(v));
        dv[ (g .< 0) .& (UB .<  Inf) ] .= -1;
        dv[ (g .> 0) .& (LB .> -Inf) ] .=  1;

    return v, dv

end

function chooseStep(x, H_hat, g_hat, gn, gn_hat, D, trust_radius, theta, LB, UB)

    # Ok so we found a step p that minimizes the quadratic approximation thingy and lies within the trust region.
    # If x + p also lies within the feasible region then this is the step we are going to take

        if withinBounds(x + gn, LB, UB)
            step        = gn;
            step_hat    = gn_hat;
            step_value  = evaluateQuadratic(H_hat, g_hat, gn_hat);
            println("          The Inexact Newton step was chosen")
            return step, step_hat, step_value
        end

    # If x + p does not lie within the feasible region,
    # we consider THREE different steps and take the best one

    # The steps should lie inside the trust region and inside the
    # feasible region. To check whether the steps lie inside the trust
    # region we use the _hat variables (then we can check their
    # 2-norms, otherwise we would have to mess with norms). To check
    # whether steps lie in the feasible region we work in the original
    # variables

    # POTENTIAL STEP 1: Move from x in direction p until you hit the feasible region boundary.
    #                   From there move in reflected direction rf
    p_steplength, boundary_hit = distanceToFeasibleRegion(x, gn, LB, UB);

    # Reflected direction (it doesn't matter if we do the
    # reflection on rf_hat or on rf since the scaling thing is
    # diagonal)
    rf_hat = gn_hat;
    rf_hat[boundary_hit] = -1 * rf_hat[boundary_hit];
    rf = D .* rf_hat;

    # Add p_steplength * p to x so that we are sitting on the boundary
    gn             = p_steplength * gn;
    gn_hat         = p_steplength * gn_hat;
    x_on_boundary = x + gn;

    # From (x + p), which lies on the boundary, we - at most - move in the direction r until
    # we hit either the trust region boundary or the feasible region boundary
    meh, to_trust = distanceToTrustRegion2(gn_hat, rf_hat, trust_radius);
    to_feasible, meh = distanceToFeasibleRegion(x_on_boundary, rf, LB, UB);

    # Find lower and upper bounds on a step size  along the reflected
    # direction, considering the strict feasibility requirement. There is no
    # single correct way to do that, the chosen approach seems to work best
    # on test problems.
    rf_steplength = min(to_trust, to_feasible);

    if rf_steplength > 0

        rf_steplength_l = (1 - theta) * p_steplength / rf_steplength; # We need to be in the interior, hence the (1-theta) to move just a little bit away from the boundary

        if rf_steplength == to_feasible
            rf_steplength_u = theta * to_feasible; # Multiply by theta < 1 to stay in the interior
        elseif rf_steplength == to_trust
            rf_steplength_u = to_trust;
        end

    else
        # Wtf is this
        println("        rf_steplength <= 0? What's going on?")
        rf_steplength_l = 0;
        rf_steplength_u = -1;
    end

    # Check if reflection step is available.
    if rf_steplength_l <= rf_steplength_u
        a, b, c = buildQuadratic1D(H_hat, g_hat, rf_hat, gn_hat);
        rf_steplength, rf_value = minimizeQuadratic1D(a, b, rf_steplength_l, rf_steplength_u, c);
        rf_hat = rf_hat * rf_steplength;
        rf_hat = rf_hat + gn_hat;
        rf = D .* rf_hat;
    else
        rf_value = Inf; # Reflection step is bad in this case
    end

    # POTENTIAL STEP 2: x + theta * p (strictly interior point)

    # gn was previously scaled such that x + gn lies on the
    # boundary of the feasible region. For differentiability of
    # things we prefer to remain in the interior of the feasible
    # region. To this end, scale by some theta that is almost 1
    gn       = theta * gn;
    gn_hat   = theta * gn_hat;
    gn_value  = evaluateQuadratic(H_hat, g_hat, gn_hat);

    # POTENTIAL STEP 3: The steepest descent direction

    sd_hat = -g_hat;
    sd = D .* sd_hat;

    to_trust         = trust_radius / norm(sd_hat);
    to_feasible, meh = distanceToFeasibleRegion(x, sd, LB, UB);

    if to_feasible < to_trust
        sd_steplength_max = theta * to_feasible;
    else
        sd_steplength_max = to_trust;
    end

    a, b, c = buildQuadratic1D(H_hat, g_hat, sd_hat, zeros(length(g_hat)));
    sd_steplength, sd_value = minimizeQuadratic1D(a, b, 0, sd_steplength_max,0);
    sd_hat = sd_steplength * sd_hat;
    sd     = sd_steplength * sd;

    # Now choose the one that gives the smallest value

    values = [gn_value rf_value sd_value]
    minVal = minimum(values);
    index = findfirst(y -> y == minVal, values)
    index = index[1]
    println("        gn: $(gn_value), rf: $(rf_value), sd: $(sd_value)")

        if index == 1
            step        = gn;
            step_hat    = gn_hat;
            step_value  = gn_value;
            println("        The Inexact Newton step, restricted to feasible region, was chosen")
        end
        if index == 2
            step        = rf;
            step_hat    = rf_hat;
            step_value  = rf_value;
            println("        The Reflected Inexact Newton step was chosen")
        end
        if index == 3
            step        = sd;
            step_hat    = sd_hat;
            step_value  = sd_value;
            println("        The Steepest Descent step was chosen")
        end

    return step, step_hat, step_value

end

function evaluateQuadratic(H, g, s)
    value = 0.5 * s' * H(s) + g' * s
    return value
end

function buildQuadratic1D(H, g, s, s0)
    # Parameterize a multivariate quadratic function along a line.
    # f(t) = 0.5 * (s0 + s*t)' * H * (s0 + s*t) + g' * (s0 + s*t)

    a = 0.5 * s' * H(s)
    b = g' * s + 0.5 * s' * H(s0) + 0.5 * s' * H(s0)
    c = g' * s0 + 0.5 * s0' * H(s0);

    return a,b,c

end

function minimizeQuadratic1D(a, b, lb, ub, c)
    # Minimize a 1-d quadratic function subject to bounds:
    # min (a * t.^2 + b * t + c) s.t. lb <= t <= ub

    # Bounds must be finite.

    # The minimum is either found on the boundary, or at the point where
    # the gradient vanishes (-0.5b/a)
    t = [lb, ub];
    if a != 0
        extremum = -0.5 * b / a;
        if ((lb < extremum) && (extremum < ub))
            t = [lb, extremum, ub];
        end
    end

    y = a .* t.^2 .+ b .* t .+ c;
    minval = minimum(y);
    index = findfirst(x -> x == minval, y)
    argument = t[index[1]];

    return argument, minval

end

function withinBounds(parameters, LB, UB)
    allParametersWithinBounds = all( (parameters .>= LB) .& (parameters .<= UB) );

    return allParametersWithinBounds
end

function distanceToTrustRegion2(x, s, trust_radius)

    # Find the intersection of a line with the boundary of a trust region.
    # This function solves the quadratic equation with respect to t:
    # ||(x + s*t)||^2 = trust_radius^2.
    # Returns [t-, t+], the negative and positive roots
    # Raises ValueError if `s` is zero or `x` is not within the trust region.

    a = s' * s;

        if a == 0
            println("    distanceToTrustRegion2: WTF s is zero")
            return
        end

    b = x' * s;
    c = x' * x - trust_radius^2;

        if c > 0
            println("     distanceToTrustRegion2: WTF")
            return
        end

    d = sqrt(b*b - a*c);  # Root from one fourth of the discriminant.

    # Computations below avoid loss of significance, see "Numerical Recipes".
    q = -(b + abs(d) * sign(b));
    t1 = q / a;
    t2 = c / q;

    if t1 < t2
        t_negative = t1;
        t_positive = t2;
    else
        t_negative = t2;
        t_positive = t1;
    end

    return t_negative, t_positive

end