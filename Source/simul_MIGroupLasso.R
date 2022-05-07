AR1 <- function(tau, m) {
  if(m==1) {R <- 1}
  if(m > 1) {
    R <- diag(1, m)
    for(i in 1:(m-1)) {
      for(j in (i+1):m) {
        R[i,j] <- R[j,i] <- tau^(abs(i-j))
      }
    }
  }
  return(R)
}

#' function for standardize covariate
normalize = function(x){
  y = sqrt(length(x))*(x-mean(x))/sqrt(sum((x-mean(x))^2))
  return(y)
}

simul  <- function(N = 1000, p = 500, p_true = 5, m1 = 100, cor = 0.3, mag = c(0.5,1)){
  Sigma_z1    <- diag(p)
  Corr1       <- AR1(cor,m1)
  diag(Corr1) <- 1
  z           <- NULL
  j           <- 0
  
  #Simulate z
  while(j<(p/m1)){
    j    <- j+1
    z    <- cbind(z,rmvnorm(N, mean=rep(0,m1), sigma=Corr1))
  }
  z      <- apply(z,2,normalize)
  
  #Simulate True beta
  TrueBeta       <- rep(0, p)
  TrueBeta_index <- sample(1:p,p_true,replace=FALSE)
  signbeta       <- sample(c(-1,1),p_true,replace=T)
  #mag            <- runif(p_true, 1,2)
  mag            <- runif(p_true, mag[1], mag[2])
  TrueBeta[TrueBeta_index]  <- mag*signbeta
  xbeta          <- z%*%TrueBeta
  U              <- runif(N, 0, 1)
  #Simulate the time and death indicator
  pre_time       <- -log(U)/(1*exp(xbeta))
  pre_censoring  <- runif(N,0,3)
  pre_censoring  <- pre_censoring*(pre_censoring<3)+3*(pre_censoring>=3)
  tcens          <- (pre_censoring<pre_time) # censoring indicator
  delta          <- 1-tcens
  time           <- pre_time*(delta==1)+pre_censoring*(delta==0)
  #order delta, z and time by time.
  delta          <- delta[order(time)]
  z              <- z[order(time),]
  time           <- time[order(time)]
  
  return( list(delta=delta,  z =z, time = time, p=p, N = N, TrueBeta = TrueBeta) )
  
}
