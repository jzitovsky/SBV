# libraries to run functions and simulations
library(abind)   
library(MASS)  
library(dplyr, warn.conflicts=F)
library(pracma, warn.conflicts=F)
library(caret, warn.conflicts=F)

#simulation helper functions
norm_vec = function(x) sqrt(sum(x^2))
expit = function (x) 1/(1+exp(-x))
sigmoid = function(x) 1/(1+exp(-x))
linearT = function(S) cbind(1,S)
quadratic = function(S) cbind(1, model.matrix(~-1+.^2, data=as.data.frame(S)), S^2)
cubic = function(S) cbind(1, model.matrix(~-1+.^3, data=as.data.frame(S))[,-c(1:ncol(S))], model.matrix(~-1+I(S^2)*S))
arbPoly = function(S, degree=4) {
  if (degree==1) { 
    return(linearT(S))
  } else if (degree==2) { 
    return(quadratic(S))
  } else if (degree==3) { 
    return(cubic(S))
  } else {
    return(cbind(1, poly(S, degree=degree, raw=T, simple=T)))
  }
}
select = dplyr::select 



#generate ultra simple simulation simulation
genmodS = function(n, capT, x, custom=F,
                   policyFun=NULL, randomInit=F) {
  capT = capT + 1
  S = array(NA, dim = c(n, 4, capT))
  A = U = matrix(NA, nrow = n, ncol = capT)
  S[,,1] = cbind(rnorm(n), rnorm(n), rnorm(n), rnorm(n))

  for (t in 2:capT) {
    if (custom & !(t==2 & randomInit)) {
      A[,t-1] = as.numeric(as.character(policyFun(S[,,t-1])))
    } else {
      A[,t-1] = rbinom(n, 1, prob = 0.5)
    }
    S[,1,t] = sqrt(x)*S[,1,t-1] + (A[,t-1]-0.5) + sqrt(0.75-x)*rnorm(n, 0, 1)
    S[,2,t] = sqrt(4*x-2)*S[,2,t-1] + sqrt(3-4*x)*rnorm(n,0,1)
    S[,3,t] = sqrt(4*x-2)*S[,3,t-1] + sqrt(3-4*x)*rnorm(n,0,1)
    S[,4,t] = sqrt(4*x-2)*S[,4,t-1] + sqrt(3-4*x)*rnorm(n,0,1)
    U[,t-1] = S[,1,t]
  }
  return(list(S = S, A = A, U = U))
}


#get data from 3-dimmensional arrays into long data frame or matrix
getLongData = function(Phi, A, U, mu = function(x) 0.5, returnMatrix=T, makeIntercept=F) {
  n = dim(Phi)[1]
  capT = dim(Phi)[3]-1
  p = dim(Phi)[2]
  PhiMat = aperm(Phi, perm = c(3,1,2)) #BULK. Further improvements can be made by returning this permutation to begin with from the simulation
  dim(PhiMat) = c(n*(capT+1),p)
  UVec = as.vector(t(U))
  AVec = as.vector(t(A))
  propensities = mu(AVec)
  i = rep(1:n, each=capT+1)
  t = rep(1:(capT+1), n)
  if (makeIntercept) {
    longData = cbind(1, PhiMat, AVec, UVec, i, t, propensities)
    p=p+1
  } else {
    longData = cbind(PhiMat, AVec, UVec, i, t, propensities)
  }
  start = ifelse(makeIntercept, 2, 1)
  ind = apply(as.matrix(longData[,start:(ncol(PhiMat)+start-1)]), 1, function(x) all(is.na(x)))
  longData = longData[!ind, ]
  colnames(longData) = c(paste0("S",1:p),"A","U",'i','t', 'propensities')
  if (!returnMatrix) longData = as.data.frame(longData)
  return(longData)
}


#Estimate Policy by Linear FQI, full-regularization
QCompsRidge = function(simData, tolerance=1e-4, maxiter=200, transform=linearT, gamma=0.9, lambda=0, 
                         P=NULL, terminal=NULL, ...) {
  longData = getLongData(Phi = simData$S, A = simData$A, U = simData$U, makeIntercept = F, returnMatrix = F)
  t = longData$t
  n = max(longData$i)
  m = nrow(longData)
  finalTime = is.na(longData$U)
  U = longData$U[!finalTime]
  AC = as.factor(longData$A[!finalTime])
  actionSpace = sort(unique(AC))
  Phi = longData %>% select(-c(A,U,i,t,propensities)) %>% as.matrix() %>% transform(...)
  PhiC = Phi[!finalTime,]
  PhiN = Phi[t>1,]
  rm(Phi)
  X = model.matrix(~PhiC[,-c(1)]+AC*PhiC[,-c(1)])
  if (is.null(terminal)) terminal = rep(0, nrow(longData))
  if (length(terminal)==1) if (terminal=='auto') terminal = as.numeric(finalTime)
  terminalC = terminal[!finalTime]
  terminalN = terminal[t>1]
  
  if (is.null(P)) {
    scales = apply(X[,-c(1)], 2, sd)
    P = diag(c(0, scales^2))
    plier = ncol(PhiC)
  }
  Hmat = solve(crossprod(X)/m+lambda*P)%*%t(X)/m 
  
  iter=1
  eps=Inf
  thetaOld = theta = rep(0, ncol(PhiC)*length(actionSpace))
  while (eps>tolerance & iter<=maxiter) {
    QArrow = list()
    for (i in 1:length(actionSpace)) {
      A2 = factor(rep(actionSpace[i], nrow(PhiC)), levels=levels(AC), labels=levels(AC))
      X2 = model.matrix(~PhiN[,-c(1)]+A2*PhiN[,-c(1)])
      QArrow[[i]] = (1-terminalN)*(X2%*%theta)
    }
    QMax = rep(-Inf, nrow(PhiC))
    for (i in 1:length(QArrow)) QMax = pmax(QMax, QArrow[[i]])
    Bellman = U + gamma*QMax
    theta = Hmat%*%Bellman
    eps = norm_vec(abs(theta-thetaOld))
    thetaOld = theta
    iter = iter+1
  } 
  wDec = theta[(ncol(PhiN)+1):(2*ncol(PhiN))]
  
  QFun = function(Phi,A, terminal2=NULL) {
    if (is.null(terminal2)) terminal2 = rep(0, nrow(Phi))
    if (ncol(Phi)!=ncol(PhiC)) Phi=transform(Phi,...)
    if (length(A)==1) A = rep(A, nrow(Phi))
    A = factor(A, levels = levels(AC), labels = levels(AC))
    if (nrow(Phi)>1) X = model.matrix(~Phi[,-c(1)]+A*Phi[,-c(1)])
    if (nrow(Phi)==1) {
      Phi2 = rbind(Phi, rep(0, ncol(Phi)))
      A2 = factor(c(as.vector(A), '1'), levels = levels(AC), labels = levels(AC))
      X2 = model.matrix(~Phi2[,-c(1)]+A2*Phi2[,-c(1)])
      X = t(as.matrix(X2[1,]))
    }
    (1-terminal2)*(X%*%theta)
  }
  
  pi = function(S, terminal2=NULL) {
    QArrow = matrix(ncol=length(actionSpace), nrow=nrow(S))
    for (i in 1:length(actionSpace)) QArrow[,i] = QFun(S, actionSpace[i], terminal2)
    return(actionSpace[max.col(QArrow)])
  } 
  
  QMaxN = rep(-Inf, nrow(PhiN))
  for (i in 1:length(actionSpace)) QMaxN = pmax(QMaxN, QFun(PhiN, actionSpace[i], terminalN))
  criterion = max(abs(t(X)%*%(U + gamma*QMaxN - QFun(PhiC, AC, terminalC))-m*lambda*P%*%theta))
  return(list(pi=pi, QFun=QFun, theta=theta, eps=eps, criterion=criterion, iter=iter))
}


#evaluate given linear policy
evalBeta = function(n = 1000, capT = 100, seed=NULL, ...) {
  set.seed(seed)
  data = genmodS(n = n, capT = capT, ...)
  value = mean(data$U, na.rm = T)
  return(value)
}

