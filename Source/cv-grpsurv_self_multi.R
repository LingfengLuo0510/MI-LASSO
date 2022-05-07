
cv.grpsurv_self_multi <- function(Xlist, delta, time, group, ..., nfolds=10, seed, fold, se=c('quick', 'bootstrap'), 
                                  cv.method = c('LinPred', 'VVH'),
                                  returnY=FALSE, trace=FALSE) {
  
  se <- match.arg(se)
  
  # Complete data fit
  fit.args <- list(...)
  fit.args$Xlist <- Xlist
  fit.args$delta <- delta
  fit.args$time <- time
  fit.args$group <- c(1:p)
  fit.args$K <- length(Xlist)
  
  
  fit.args$returnX <- TRUE
  #fit <- do.call("grpsurv", fit.args)
  fit <- do.call("grpsurv_self_multi", fit.args)
  # Get standardized X, y
  X <- fit$XX
  delta_matrix <- fit$delta_matrix
  time_matrix <- fit$time_matrix
  returnX <- list()$returnX
  if (is.null(returnX) || !returnX) fit$X <- NULL
  
  # Set up folds
  n <- nrow(X)
  #if (!missing(seed)) set.seed(seed)
  #if (missing(fold)) {
  
  ind1 <- which(fit$delta==1)
  ind0 <- which(fit$delta==0)
  n1 <- length(ind1)
  n0 <- length(ind0)
  fold1 <- 1:n1 %% nfolds
  fold0 <- (n1 + 1:n0) %% nfolds
  fold1[fold1==0] <- nfolds
  fold0[fold0==0] <- nfolds
  fold_sub <- integer(n/K)
  fold <- integer(n) 
  fold_sub[fit$delta==1] <- sample(fold1)
  fold_sub[fit$delta==0] <- sample(fold0)
  fold <- rep(fold_sub, each = K)
  #} else {
  #  nfolds <- max(fold)
  #}
  Y <- matrix(NA, nrow=n, ncol=length(fit$lambda))
  
  cv.args <- list()
  cv.args$lambda <- fit$lambda
  cv.args$group <- fit$group
  #cv.args$group.multiplier <- fit$
  cv.args$warn <- FALSE
  
  if(cv.method == 'LinPred'){
    for (i in 1:nfolds) {
      res <- cvf.surv(i, Xlist, fit$delta, fit$time, K, fold, fold_sub, cv.args)
      Y[fold==i, 1:res$nl] <- res$yhat
    }
    ind <- which(apply(is.finite(Y), 2, all))
    lambda <- fit$lambda[ind]
    L <- loss.grpsurv_self_multi(delta_matrix, Y, total=FALSE)
    cve <- apply(L, 2, sum)/sum(delta_matrix)
    # cvse <- apply(L, 2, sd)*sqrt(nrow(L))/sum(delta_matrix)
  } else if (cv.method == 'VVH'){
    lambda  <- cv.args$lambda
    cve     <- rep(0, length(lambda))
    for (i in 1:nfolds) {
      res                   <- cvf.survVVH(i, Xlist, fit$delta, fit$time, K, fold, fold_sub, cv.args)
      EtaAll                <- res$yhatall
      EtaK                  <- res$yhat
      L_K                   <- loss.grpsurv_self_multi(delta_matrix[fold!=i], EtaK, K, total=FALSE)
      L_all                 <- loss.grpsurv_self_multi(fit$delta, EtaAll, K, total=FALSE)
      cve                   <- cve +  (apply(L_all, 2, sum)/sum(delta_matrix) - apply(L_K, 2, sum)/sum(delta_matrix))
    }
  }
  
  
  
  min <- which.min(cve)
  
  
  val <- list(cve=cve, fold=fold, lambda=lambda, fit=fit, min=min, lambda.min=lambda[min], null.dev=cve[1])
  if (returnY) val$Y <- Y
  structure(val, class=c("cv.grpsurv", "cv.grpreg"))
  
}


cvf.surv <- function(i, XX, delta, time, K, fold, fold_sub, cv.args) {
  Xlist_temp <- NULL
  for (k in 1:K) {
    Xlist_temp[[k]] <- XX[[k]][fold_sub!=i, , drop=FALSE]
  }
  cv.args$Xlist <- Xlist_temp
  cv.args$time <- time[fold_sub!=i]
  cv.args$delta <- delta[fold_sub!=i]
  cv.args$K <- K
  fit.i <- do.call("grpsurv_self_multi", cv.args)
  
  Xlist_temp2 <- NULL
  for (k in 1:K) {
    Xlist_temp2[[k]] <- XX[[k]][fold_sub==i, , drop=FALSE]
  }
  
  nl <- length(fit.i$lambda)
  yhat <- predict.grpsurv_self_multi(fit.i, Xlist_temp2, K)
  
  list(nl=length(fit.i$lambda), yhat=yhat)#, loss=loss)
}


cvf.survVVH <- function(i, XX, delta, time, K, fold, fold_sub, cv.args) {
  Xlist_temp <- NULL
  for (k in 1:K) {
    Xlist_temp[[k]] <- XX[[k]][fold_sub!=i, , drop=FALSE]
  }
  cv.args$Xlist <- Xlist_temp
  cv.args$time <- time[fold_sub!=i]
  cv.args$delta <- delta[fold_sub!=i]
  cv.args$K <- K
  fit.i <- do.call("grpsurv_self_multi", cv.args)
  
  Xlist_temp2 <- NULL
  for (k in 1:K) {
    Xlist_temp2[[k]] <- XX[[k]][fold_sub!=i, , drop=FALSE]
  }
  
  nl        <- length(fit.i$lambda)
  yhat      <- predict.grpsurv_self_multi(fit.i, Xlist_temp2, K)
  yhatall   <- predict.grpsurv_self_multi(fit.i, XX, K)
  
  list(nl=length(fit.i$lambda), yhat=yhat, yhatall=yhatall)#, loss=loss)
}
