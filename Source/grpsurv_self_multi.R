grpsurv_self_multi <- function(Xlist, delta, time, group, K, penalty="grLasso",
                               alpha=1, nlambda=100, lambda,
                               lambda.min=0.001, eps=.001, max.iter=10000,
                               dfmax=p, gmax=length(unique(group)), tau=1/3, group.multiplier,
                               warn=TRUE, returnX=FALSE, ...) {
  
  #check
  if (nlambda < 2) stop("nlambda must be at least 2", call.=FALSE)
  if (alpha > 1 | alpha <= 0) stop("alpha must be in (0, 1]", call.=FALSE)
  
  gamma = 3
  p = ncol(Xlist[[1]])
  n = length(delta)
  # Construct XG, Y
  bilevel <- FALSE
  
  #Y <- newS(y)
  #XG <- newXG(X[Y$ind,], group, group.multiplier)
  delta_matrix <- matrix(delta, length(delta), K)
  paste("Y", 1:ncol(delta_matrix), sep="")
  attributes(delta_matrix) <- NULL
  
  time_matrix <- matrix(time,length(time), K)
  paste("Y", 1:ncol(time_matrix), sep="")
  attributes(time_matrix) <- NULL
  
  group <- multiG(c(1:p),K)[-c(1:(K-1))]

  if(K>1){
    m = K
    A <- matrix(0, m*n, m*p)
    for (i in 1:m) {
      z = as.matrix(Xlist[[i]])
      A[m*(1:n)-i+1, m*(1:p)-i+1] <- z
    }
    #X <- cbind(matrix(as.double(diag(m)), m*n, m, byrow=TRUE)[,2:m], A)
    X <- cbind(A)
  } else{
    X <- Xlist[[1]]
  }
  std <- standardize(X)
  XX <- std[[1]]
  center <- as.vector(std[[2]])
  scale <- std[[3]]
  # XX <- X
  
  #XX <- orthogonalize(XX, group)
  group.multiplier <- rep(rep(sqrt(K),2),p)

  #if (nrow(XG$X) != length(Y$fail)) stop("X and y do not have the same number of observations", call.=FALSE)

  # Set up lambda
  if (missing(lambda)) {
    lambda <- setupLambdaCox_self(XX, time_matrix, delta_matrix, group, penalty, alpha, lambda.min, nlambda, group.multiplier)
    user.lambda <- FALSE
  } else {
    nlambda <- length(lambda)
    user.lambda <- TRUE
  }
  
  #fit
  K <- as.integer(table(group))
  K0 <- as.integer(if (min(group)==0) K[1] else 0)                     # =0 if each variable has a group index
  K1 <- as.integer(if (min(group)==0) cumsum(K) else c(0, cumsum(K)))  
  dfmax <- p
  warn  <- TRUE
  res   <- gdfit_cox(XX, delta_matrix, penalty, K1, K0, lambda, alpha, eps = eps, max_iter = as.integer(max.iter),
                     as.double(gamma), group.multiplier, as.integer(dfmax), as.integer(gmax), as.integer(warn), as.integer(user.lambda))
  # res2  <- .Call("gdfit_cox", XG$X, Y$fail, penalty, K1, K0, lambda, alpha, eps, as.integer(max.iter),
  #                          as.double(gamma), XG$m, as.integer(dfmax), as.integer(gmax), as.integer(warn), as.integer(user.lambda))
  # 
  
  b     <- matrix(t(res$beta),p*K,nlambda)
  iter  <- as.vector(res$iter)
  df    <- as.vector(res$df)
  loss  <- -1*as.vector(res$Loss)
  Eta   <- matrix(t(res$Eta),n,nlambda)
  
  ind <- !is.na(iter)
  b <- b[, ind, drop=FALSE]
  iter <- iter[ind]
  lambda <- lambda[ind]
  df <- df[ind]
  loss <- loss[ind]
  #if (iter[1] == max.iter) stop("Algorithm failed to converge for any values of lambda.  This indicates a combination of (a) an ill-conditioned feature matrix X and (b) insufficient penalization.  You must fix one or the other for your model to be identifiable.", call.=FALSE)
  if (warn & any(iter==max.iter)) warning("Algorithm failed to converge for all values of lambda", call.=FALSE)
  
  
  # Unstandardize
  #print(group)
  #print(dim(b))
  #print(dim(XX))
  #b <- unorthogonalize(b, XX, group, intercept=FALSE)
  #if (XG$reorder) b <- b[XG$ord.inv,]
  #beta <- matrix(0, nrow=length(scale), ncol=ncol(b))
  #beta[XG$nz,] <- b / XG$scale[XG$nz]
  
  #dimnames(beta) <- list(XG$names, round(lambda, digits=4))
  
  # Output
  val <- structure(list(beta = b,
                        group = group,
                        lambda = lambda,
                        penalty = penalty,
                        gamma = gamma,
                        alpha = alpha,
                        loss = loss,
                        n = n,
                        df = df,
                        iter = iter,
                        group.multiplier = group.multiplier,
                        time = time,
                        delta = delta,
                        #fail = Y$fail,
                        W = exp(Eta)),
                        class = c("grpsurv", "grpreg"))
  if (returnX) {
    val$XX <- XX
    val$delta_matrix <- delta_matrix
    val$time_matrix <- time_matrix
  }
  val
  
}

# library(grpreg)
# data(Lung)
# X <- Lung$X
# y <- Lung$y
# group <- Lung$group
# 
# ##
# fit  <- grpsurv(X, y, group, returnX = TRUE)
# 
# ##
# fit2 <- grpsurv_self(X, y, group, returnX = TRUE)
# 
# plot(fit)
# plot(fit2)
# 
# identical(fit,fit2)
# 
# identical(fit2$lambda, fit$lambda)
# 
# identical(fit2$beta, fit$beta)
# 
# identical(fit2$iter, fit$iter)
# 
# identical(fit2$df, fit$df)
# 
# identical(fit2$W, fit$W)
# 
# identical(fit2$time, fit$time)
# identical(fit2$fail, fit$fail)

##################
# source("cv-grpsurv_self.R")
# 
# 
# library(grpreg)
# data(Lung)
# X <- Lung$X
# y <- Lung$y
# group <- Lung$group
# 
# set.seed(1)
# fit1 <- cv.grpsurv_self(Lung$X, Lung$y,  Lung$group, se = "quick")
# set.seed(1)
# fit2 <- cv.grpsurv(Lung$X, Lung$y,  Lung$group, se = "quick")
# 
# fit1$fold
# fit2$fold
# 
# fit1$min
# fit2$min





