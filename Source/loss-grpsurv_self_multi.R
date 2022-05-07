# 
# loss.MIGroupLasso <- function(y, eta, K, total=TRUE) {
#   l <- dim(eta)[2]
#   ind <- order(y[,1])
#   d <- as.integer(y[ind,2])
#   N <- length(d)
#   ind2 <- NULL
#   for(i in 1:K){
#     ind2 <- c(ind2, ind + N*(i-1))
#   }
#   
#   eta <- eta[ind2, , drop=FALSE]
#   r   <- matrix(0,N*K,l)
#   for(i in 1:K){
#     samplei <- (N*(i-1)+1) : (N*i)
#     r[samplei,] <- apply(eta[samplei,], 2, function(x) rev(cumsum(rev(exp(x)))))
#   }
#   
#   loglik = 0
#   for (i in 1:K) {
#     samplei  <- (N*(i-1)+1) : (N*i) & d==1
#     loglik = loglik -2*(eta[samplei, , drop=FALSE] - log(r)[samplei, , drop=FALSE])
#   }
#   
#   return(loglik)
# }

loss.grpsurv_self_multi <- function(delta, eta, K, total=TRUE) {
  l <- dim(eta)[2]
  #ind <- order(y[,1])
  #print(ind)
  d <- as.integer(delta)
  if (is.matrix(eta)) {
    eta <- eta[, , drop=FALSE]
    r <- apply(eta, 2, function(x) rev(cumsum(rev(exp(x)))))
  } else {
    eta <- as.matrix(eta[ind])
    r <- as.matrix(rev(cumsum(rev(exp(eta)))))
  }
  if (total) {
    return(-2*(crossprod(d, eta) - crossprod(d, log(r))))
  } else {
    return(-2*(eta[d==1, , drop=FALSE] - log(r)[d==1, , drop=FALSE]))
  }
  
  
  
  return(loglik)
}
