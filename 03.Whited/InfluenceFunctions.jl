"""
    InfluenceFunctions

Conditional moments, skewness, and their influence functions.
Port of influencefunctions.R. All influence functions return an n×1 matrix,
except `beta_infl`, which returns n×k. Estimators return scalars, except
`beta`, which returns a k×1 matrix.
"""
module InfluenceFunctions

using LinearAlgebra
using Statistics

export mean_cond, cov_cond, beta, skew, skew_cond,
       mean_infl, beta_infl, cov_infl, mean_cond_infl,
       mean_ratio_infl, cov_cond_infl, skew_infl, skew_cond_infl

_vec(a::AbstractVector) = float.(a)
_vec(a::AbstractMatrix) = float.(vec(a))
_col(a) = reshape(_vec(a), :, 1)
_mat(a::AbstractVector) = reshape(float.(a), :, 1)
_mat(a::AbstractMatrix) = float.(a)

# Conditional mean and covariance ---------------------------------------------

"""E(x | dum = 1), dum ∈ {0, 1}."""
function mean_cond(x, dum)
    x, dum = _vec(x), _vec(dum)
    return mean(x .* dum) / mean(dum)
end

"""cov(x, y | dum = 1), dum ∈ {0, 1}."""
function cov_cond(x, y, dum)
    x, y, dum = _vec(x), _vec(y), _vec(dum)
    μ_dum = mean(dum)
    return mean(x .* y .* dum) / μ_dum -
           (mean(x .* dum) / μ_dum) * (mean(y .* dum) / μ_dum)
end

"""Slope of the linear regression y = x * β + u."""
function beta(x, y)
    X, yv = _mat(x), _col(y)
    return (X' * X) \ (X' * yv)
end

"""E[(x - E[x])^3] / E[(x - E[x])^2]^(3/2)."""
function skew(x)
    x = _vec(x)
    d = x .- mean(x)
    return mean(d .^ 3) / mean(d .^ 2)^1.5
end

"""Conditional skewness, dum ∈ {0, 1}."""
function skew_cond(x, dum)
    x, dum = _vec(x), _vec(dum)
    μ_dum = mean(dum)
    μc_x3 = mean(x .^ 3 .* dum) / μ_dum
    μc_x2 = mean(x .^ 2 .* dum) / μ_dum
    μc_x = mean(x .* dum) / μ_dum
    F_dum = μc_x3 - 3 * μc_x2 * μc_x + 2 * μc_x^3
    G_dum = μc_x2 - μc_x^2
    return F_dum / G_dum^1.5
end

# Influence functions ---------------------------------------------------------

"""Influence function for the mean."""
function mean_infl(x)
    x = _vec(x)
    return _col(x .- mean(x))
end

"""Influence function for OLS coefficients in y = x * β + u."""
function beta_infl(x, y)
    X, yv = _mat(x), _vec(y)
    n = length(yv)
    b = (X' * X) \ (X' * yv)
    u_hat = yv .- X * b
    # infl = (X .* u_hat) * inv(X'X / n)
    return (X .* u_hat) / ((X' * X) ./ n)
end

"""Influence function for cov(x, y)."""
function cov_infl(x, y)
    x, y = _vec(x), _vec(y)
    μ_xy, μ_x, μ_y = mean(x .* y), mean(x), mean(y)
    return _col(x .* y .- μ_xy .- μ_y .* (x .- μ_x) .- μ_x .* (y .- μ_y))
end

"""Influence function for E(x | dum = 1), dum ∈ {0, 1}."""
function mean_cond_infl(x, dum)
    x, dum = _vec(x), _vec(dum)
    μ_dum = mean(dum)
    μ_x_dum = mean(x .* dum)
    return _col((μ_dum .* (x .* dum .- μ_x_dum) .-
                 μ_x_dum .* (dum .- μ_dum)) ./ μ_dum^2)
end

"""Influence function for E(x) / E(y)."""
function mean_ratio_infl(x, y)
    x, y = _vec(x), _vec(y)
    μ_x, μ_y = mean(x), mean(y)
    return _col((μ_y .* (x .- μ_x) .- μ_x .* (y .- μ_y)) ./ μ_y^2)
end

"""Influence function for cov(x, y | dum = 1), dum ∈ {0, 1}."""
function cov_cond_infl(x, y, dum)
    x, y, dum = _vec(x), _vec(y), _vec(dum)
    μ_dum = mean(dum)
    μ_x_y_dum = mean(x .* y .* dum)
    μ_x_dum = mean(x .* dum)
    μ_y_dum = mean(y .* dum)
    infl = (μ_dum .* (x .* y .* dum .- μ_x_y_dum) .-
            μ_x_y_dum .* (dum .- μ_dum)) ./ μ_dum^2 .-
           μ_y_dum / μ_dum .* (μ_dum .* (x .* dum .- μ_x_dum) .-
            μ_x_dum .* (dum .- μ_dum)) ./ μ_dum^2 .-
           μ_x_dum / μ_dum .* (μ_dum .* (y .* dum .- μ_y_dum) .-
            μ_y_dum .* (dum .- μ_dum)) ./ μ_dum^2
    return _col(infl)
end

"""Influence function for skewness."""
function skew_infl(x)
    x = _vec(x)

    μ_x3 = mean(x .^ 3)
    μ_x2 = mean(x .^ 2)
    μ_x = mean(x)
    F = μ_x3 - 3 * μ_x2 * μ_x + 2 * μ_x^3
    G = μ_x2 - μ_x^2

    ψ_x = x .- μ_x
    ψ_x2 = x .^ 2 .- μ_x2
    ψ_x3 = x .^ 3 .- μ_x3
    ψ_F = ψ_x3 .- 3 .* (ψ_x2 .* μ_x .+ μ_x2 .* ψ_x) .+ 6 * μ_x^2 .* ψ_x
    ψ_G = ψ_x2 .- 2 * μ_x .* ψ_x

    return _col((ψ_F .* G^1.5 .- 1.5 * F * G^0.5 .* ψ_G) ./ G^3)
end

"""Influence function for conditional skewness, dum ∈ {0, 1}."""
function skew_cond_infl(x, dum)
    x, dum = _vec(x), _vec(dum)

    x_dum = x .* dum
    x2_dum = x .^ 2 .* dum
    x3_dum = x .^ 3 .* dum

    μ_dum = mean(dum)
    μ_x3_dum = mean(x3_dum)
    μ_x2_dum = mean(x2_dum)
    μ_x_dum = mean(x_dum)
    μc_x3 = μ_x3_dum / μ_dum
    μc_x2 = μ_x2_dum / μ_dum
    μc_x = μ_x_dum / μ_dum
    F_dum = μc_x3 - 3 * μc_x2 * μc_x + 2 * μc_x^3
    G_dum = μc_x2 - μc_x^2

    ψ_x_dum = ((x_dum .- μ_x_dum) .* μ_dum .-
               μ_x_dum .* (dum .- μ_dum)) ./ μ_dum^2
    ψ_x2_dum = ((x2_dum .- μ_x2_dum) .* μ_dum .-
                μ_x2_dum .* (dum .- μ_dum)) ./ μ_dum^2
    ψ_x3_dum = ((x3_dum .- μ_x3_dum) .* μ_dum .-
                μ_x3_dum .* (dum .- μ_dum)) ./ μ_dum^2
    ψ_F_dum = ψ_x3_dum .-
              3 .* (ψ_x2_dum .* μ_x_dum .+ μ_x2_dum .* ψ_x_dum) .+
              6 * μ_x_dum^2 .* ψ_x_dum
    ψ_G_dum = ψ_x2_dum .- 2 * μ_x_dum .* ψ_x_dum

    return _col((ψ_F_dum .* G_dum^1.5 .-
                 1.5 * F_dum * G_dum^0.5 .* ψ_G_dum) ./ G_dum^3)
end

end # module
