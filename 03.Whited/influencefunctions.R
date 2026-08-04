# Define conditional mean and covariance ----
mean_cond <- function(x, dum) {
    # Conditional mean    
    # E(x | dum = 1), dum = {0, 1}
    res <- mean(x*dum)/mean(dum)
    return(res)
}
cov_cond <- function(x, y, dum) {
    # Conditional covariance
    # cov(x, y | dum = 1), dum = {0, 1}
    res <- mean(x*y*dum)/mean(dum) - (mean(x*dum)/mean(dum)) * (mean(y*dum)/mean(dum))
    return(res)
}     
beta <- function(x, y) {
    # Returns slope in a linear regression
    # y = x \beta + u
    res <- solve(t(x) %*% x) %*% t(x) %*% y
    return(res)
}
skew <- function(x) {
    # Skewness
    # E[(x - E[x])^3]/E[(x - E[x])^2]^3/2
    res <- mean((x - mean(x))^3)/(mean((x - mean(x))^2))^(3/2)
    return(res)
}
skew_cond <- function(x, dum) {
    # Conditional skewness (see note_on_skewness)
    # E[(x - E[x])^3 | dum = 1]/E[(x - E[x])^2 | dum = 1]^3/2, dum = {0, 1}
    mu_cond_x3_dum <- mean(x^3*dum)/mean(dum)
    mu_cond_x2_dum <- mean(x^2*dum)/mean(dum)
    mu_cond_x_dum <- mean(x*dum)/mean(dum)
    F_dum <- mu_cond_x3_dum - 3*mu_cond_x2_dum*mu_cond_x_dum + 2*mu_cond_x_dum^3
    G_dum <- mu_cond_x2_dum - mu_cond_x_dum^2
    res <- F_dum/(G_dum**(3/2))
    return(res)
}
# Define influence functions ----
mean_infl <- function(x) {
    # Influence function for mean
    infl <- as.matrix(x - mean(x), ncol = 1) 
    return(infl)
}
beta_infl <- function(x, y) {
    # Influence function for OLS coefficients
    # y = x \beta + u
    
    # Convert to numeric matrices
    x <- data.matrix(x)
    y <- data.matrix(y)
    
    # Compute influence functions for beta
    n <- length(y)
    beta <- solve(t(x) %*% x) %*% t(x) %*% y
    u_hat <- y - x %*% beta
    u_hat_mtx <- matrix(u_hat, nrow = length(u_hat), ncol=ncol(x), byrow = FALSE)
    infl <- t(solve(t(x) %*% x/n) %*% t(x*u_hat_mtx))
    return(infl)
}
cov_infl <- function(x, y) {
    # Influence function for covariance
    # cov(x, y)
    mu_xy <- mean(x*y)
    mu_x <- mean(x)
    mu_y <- mean(y)
    infl <- x*y - mu_xy - mu_y*(x - mu_x) - mu_x*(y - mu_y)
    infl <- as.matrix(infl, ncol = 1) 
    return(infl)
}
mean_cond_infl <- function(x, dum) {
    # Influence function for the conditional mean
    # E(x | dum = 1) with dum = {0, 1}
    mu_dum <-  mean(dum)
    mu_x_dum <-  mean(x*dum)
    infl <- (mu_dum*(x*dum - mu_x_dum) - mu_x_dum*(dum - mu_dum))/(mu_dum^2)
    infl <- as.matrix(infl, ncol = 1) 
    return(infl)
}
mean_ratio_infl <- function(x, y) {
    # Influence function for the ratio of means
    # E(x)/E(y)
    mu_x <-  mean(x)
    mu_y <-  mean(y)
    infl <- (mu_y*(x - mu_x) - mu_x*(y - mu_y))/(mu_y^2)
    infl <- as.matrix(infl, ncol = 1) 
    return(infl)
}
cov_cond_infl <- function(x, y, dum) {
    # Influence function for the conditional covariance
    # cov(x, y | dum = 1) with dum = {0, 1}
    mu_dum <-  mean(dum)
    mu_x_y_dum <- mean(x*y*dum)
    mu_x_dum <- mean(x*dum)
    mu_y_dum <- mean(y*dum)
    infl <- ((mu_dum*(x*y*dum - mu_x_y_dum) - mu_x_y_dum*(dum - mu_dum))/(mu_dum^2)
             - mu_y_dum/mu_dum*(mu_dum*(x*dum - mu_x_dum) - mu_x_dum*(dum - mu_dum))/(mu_dum^2)
             - mu_x_dum/mu_dum*(mu_dum*(y*dum - mu_y_dum) - mu_y_dum*(dum - mu_dum))/(mu_dum^2))
    infl <- as.matrix(infl, ncol = 1) 
    return(infl)
}
skew_infl <- function(x) {
    # Influence function on skewness (see note_on_skewness)
    # E[(x - E[x])^3]/E[(x - E[x])^2]^3/2
    
    # Compute moments
    mu_x3 <- mean(x^3)
    mu_x2 <- mean(x^2)
    mu_x  <- mean(x)
    F <- mu_x3 - 3*mu_x2*mu_x + 2*mu_x^3
    G <- mu_x2 - mu_x^2
    
    # Compute basic influence functions
    psi_x <- x - mean(x)
    psi_x2 <- x^2 - mean(x^2)
    psi_x3 <- x^3 - mean(x^3)
    psi_F <- psi_x3 - 3*(psi_x2*mu_x + mu_x2*psi_x) + 6*mu_x^2*psi_x
    psi_G <- psi_x2 - 2*mu_x*psi_x
    
    infl <- (psi_F*G^(3/2) - 3/2*F*G^(1/2)*psi_G)/(G^3)
    infl <- as.matrix(infl, ncol = 1) 
    return(infl)
}
skew_cond_infl <- function(x, dum) {
    # Influence function on conditional skewness (see note_on_skewness)
    # E[(x - E[x])^3|D]/E[(x - E[x])^2|D]^3/2
    
    x_dum <- x*dum
    x2_dum <- x^2*dum
    x3_dum <- x^3*dum
    
    # Compute moments
    mu_dum <- mean(dum)
    mu_x3_dum <- mean(x3_dum)
    mu_x2_dum <- mean(x2_dum)
    mu_x_dum <- mean(x_dum)
    mu_cond_x3_dum <- mean(x^3*dum)/mean(dum)
    mu_cond_x2_dum <- mean(x^2*dum)/mean(dum)
    mu_cond_x_dum <- mean(x*dum)/mean(dum)
    F_dum <- mu_cond_x3_dum - 3*mu_cond_x2_dum*mu_cond_x_dum + 2*mu_cond_x_dum^3
    G_dum <- mu_cond_x2_dum - mu_cond_x_dum^2
    
    # Compute basic influence functions
    psi_x_dum <- ((x_dum - mu_x_dum)*mu_dum - mu_x_dum*(dum - mu_dum))/(mu_dum^2)
    psi_x2_dum <- ((x2_dum - mu_x2_dum)*mu_dum - mu_x2_dum*(dum - mu_dum))/(mu_dum^2)
    psi_x3_dum <- ((x3_dum - mu_x3_dum)*mu_dum - mu_x3_dum*(dum - mu_dum))/(mu_dum^2)
    psi_F_dum <- psi_x3_dum - 3*(psi_x2_dum*mu_x_dum + mu_x2_dum*psi_x_dum) + 6*mu_x_dum^2*psi_x_dum
    psi_G_dum <- psi_x2_dum - 2*mu_x_dum*psi_x_dum
    
    infl <- (psi_F_dum*G_dum^(3/2) - 3/2*F_dum*G_dum^(1/2)*psi_G_dum)/(G_dum^3)
    infl <- as.matrix(infl, ncol = 1) 
    return(infl)
}
