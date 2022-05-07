rm(list = ls())

library(RcppArmadillo)
library(Rcpp)
library(mvtnorm)
library(splines)
library(survival)
library(grpreg)
library(Matrix)
library(mice)
library(tidyr)



#server
sourceCpp("grpregRe.cpp")

source("orthogonalize_self.R")
source("multi_self.R")
source("setupLambdaSelf.R")
source("simul_MIGroupLasso.R")
source("grpsurv_self_multi.R")

source("predict-grpsurv_self_multi.R")
source("loss-grpsurv_self_multi.R")
source("predict-grpsurv_self_multi.R")
source("cv-grpsurv_self_multi.R")


FPFNSeSpLik=function(TrueBeta=TrueBeta,beta=beta){
  FP <- length(which(TrueBeta==0 & beta!=0))
  FN <- length(which(TrueBeta!=0 & beta==0))
  Se <- length(which(TrueBeta!=0 & beta!=0))/length(which(TrueBeta!=0))
  Sp <- length(which(TrueBeta==0 & beta==0))/length(which(TrueBeta==0))
  FDP=FP/max(length(which(beta!=0)),1)
  output=c(FP, FN, Se, Sp, FDP)
  return(output)
}

#######################################
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
#######################################

library(survival)
library(mvtnorm)

N = 500
p = 50
cor = 0.3
mag = c(0.5,1.0)
##############################################
# 1: simulation
##############################################




loop=0
FPFN <- NULL
pb = txtProgressBar(style = 3)
nloop = 100
missing_rate <- NULL
complete_case <- NULL
time_used_all <- NULL
# while (loop<nloop) {
  loop=loop+1
  set.seed(loop)
  
  s   <- simul(N = N, p = p, p_true = 5, m1 = 10, cor = cor, mag = mag)
  #censor rate
  sum(s$delta==0)/N
  
  delta = s$delta
  z     = s$z
  time  = s$time
  p     = s$p
  N     = s$N
  
  #censore rate
  sum(s$delta==1)/N
  
  #sort time
  order1         <- order(time)
  
  delta          <- delta[order1]
  z              <- z[order1,]
  time           <- time[order1]
  
  #get the Nelson-Aalen estimator
  data = as.data.frame(cbind(z,delta,time)) 
  HT <- nelsonaalen(data, timevar = time, statusvar = delta)
  if(sum(is.na(HT))>0){
    next
  }
  
  #generate missing data (MAR)############################################################################################################
  
  alpha0 <- rnorm(N, -4, 0.3)  ##5.5 for around 80% percent
  Pr_R <- matrix(0,N,p)
  for (i in 1:N) {
    for (j in 11:p){
      Pr_R[i,j] = alpha0[i] + 0.2*z[i,j-10] + 0.2*delta[i] + 0.2*HT[i]
    }
  }
  Pr_R2 <- exp(Pr_R)/(1+exp(Pr_R))
  
  Rij <- matrix(1,N,p)
  for (i in 1:N) {
    for (j in 11:p){
      Rij[i,j] = sample(c(0,1),size = 1,prob = c(Pr_R2[i,j],1-Pr_R2[i,j]))
    }
  }
  
  #missing rate (change alpha 0 to change missing rate)
  missing_rate <- c(missing_rate, ((N*p-sum(Rij))/(N*p)*100))
  
  for (i in 1:N) {
    for (j in 1:p){
      if(Rij[i,j] == 0){
        z[i,j] = NA
      }
    }
  }
  
  
  
  completecase <- 0
  complete_index <- NULL
  for (i in 1:N) {
    if (sum(is.na(z[i,])) ==0 ){
      completecase = completecase + 1
      complete_index <- c(complete_index, i)
    }
  }
  complete_case <- c(complete_case, completecase)
  completecase/N*100
  
  #4. Nelson-alon
  data = cbind(z,delta,HT)
  data_mi <- mice(data)
  mydata <- NULL
  for(i in 1:5) {
    mydata[[i]] <- complete(data_mi,i)
  }
  zz <- NULL
  for(m in 1:5){
    zz[[m]] <- as.matrix(mydata[[m]][,1:p])
  }
  
  K <- 5
  
  X <- NULL
  for (i in 1:K) {
    X[[i]] <- zz[[i]]
  }

  
  time_used <- proc.time()
  result <- cv.grpsurv_self_multi(Xlist = X, delta = delta, time = time, group = c(1:p), K = K, penalty="grLasso",
                                  alpha=1, nlambda=100,
                                  lambda.min=0.001, eps=.001, max.iter=10000,
                                  dfmax=p*K, tau=1/3, group.multiplier = NULL,
                                  warn=TRUE, returnX=FALSE,
                                  cv.method = "LinPred")
  time_used <- (proc.time() - time_used)[3]
  time_used_all <- c(time_used_all, time_used)
  
  result$min
  
  index              <- c(1:p)
  
  select_grpreg = as.numeric(which(result$fit$beta[,result$min]!=0))
  select_grpreg <- select_grpreg[seq(from = K, by = K, length.out = length(select_grpreg)/K)]/K
  
  #beta_key           <- (abs(result$beta[,20])>0)              
  group_key          <- index[select_grpreg]
  select_group_key   <- unique(group_key)     
  select_group_key2  <- rep(0,p)  
  select_group_key2[select_group_key] <- 1
  beta_grpreg        <- select_group_key2
  
  FPFN <- rbind(FPFN,FPFNSeSpLik(s$TrueBeta,beta_grpreg))  # this FPFNSeSpLik is in simul.R
  setTxtProgressBar(pb, loop / nloop)
  
# }
FPFN
round(FPFN, digits = 2)
colMeans(FPFN)
round(colMeans(FPFN), digits = 2)
round(apply(FPFN, 2, sd), digits = 2)
mean(missing_rate)

