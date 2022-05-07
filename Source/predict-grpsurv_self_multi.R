predict.grpsurv_self_multi <- function(object, X, K, 
                                        type=c("link", "response", "survival", "median", "norm", 
                                               "coefficients", "vars", "nvars", "groups", "ngroups"),
                                        lambda, which=1:length(object$lambda), ...) {
  type <- match.arg(type)
  m = K
  n <- nrow(X[[1]])
  A <- matrix(0, m*n, m*p)
  for (i in 1:m) {
    z = X[[i]]
    A[m*(1:n)-i+1, m*(1:p)-i+1] <- z 
  }
  X <- cbind(A)
  # if (!missing(lambda)) {
  #   ind <- approx(object$lambda, seq(object$lambda), lambda)$y
  #   l <- floor(ind)
  #   r <- ceiling(ind)
  #   x <- ind %% 1
  #   beta <- (1-x)*object$beta[, l, drop=FALSE] + x*object$beta[, r, drop=FALSE]
  #   colnames(beta) <- round(lambda, 4)
  # } else {
    # beta <- object$beta[, which, drop=FALSE]
  # }
  
  beta <- object$beta
  
  #eta <- matrix(0,N*K,length(object$lambda))
  eta <- X %*% beta
  
  
  return(drop(eta))
}
