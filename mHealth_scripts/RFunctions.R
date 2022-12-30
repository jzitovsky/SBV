#setting options and importing libraries
options(max.print=50)
library(dplyr, warn.conflicts=F)
library(ranger)


#evaluate average reward under policy
evalPolicy = function(policyFun, n = 1000, capT = 1000, gamma=0.9, ...) {
  onlineData = genmodH(n=n, capT=capT, policyFun=policyFun, ...)
  rewards = onlineData$U
  avgSurv = mean(apply(rewards, 1, function(x) max(which(!is.na(x)))))
  avgReturn = mean(apply(rewards, 1, function(x) sum(gamma^(0:(length(x)-1))*x, na.rm=T)))
  sumReturn = mean(apply(rewards, 1, function(x) sum(x, na.rm=T)))
  return(list(avgSurv=avgSurv, avgReturn=avgReturn, sumReturn=sumReturn))
}



#get data from 3-dimmensional arrays into long data frame or matrix
getLongData = function(Phi, A, U, mu = function(x) 0.5) {
  n = dim(Phi)[1]; capT = dim(Phi)[3]-1; p = dim(Phi)[2]
  PhiMat = aperm(Phi, perm = c(3,1,2)) 
  dim(PhiMat) = c(n*(capT+1),p)
  UVec = as.vector(t(U)); AVec = as.vector(t(A))
  i = rep(1:n, each=capT+1); t = rep(1:(capT+1), n)
  longData = cbind(PhiMat, AVec, UVec, i, t)
  missing = apply(as.matrix(longData[,1:(ncol(PhiMat))]), 1, function(x) all(is.na(x)))
  longData = longData[!missing, ]
  colnames(longData) = c(paste0("S",1:p),"A","U",'i','t')
  longData = as.data.frame(longData)
  return(longData)
}



# Estimate Policies by LSPI
linearT = function(S) cbind(1,S)
quadratic = function(S) cbind(1, model.matrix(~-1+.^2, data=as.data.frame(S)), S^2)
cubic = function(S) {
  if (nrow(S)>1) return(cbind(1, model.matrix(~-1+.^3, data=as.data.frame(S))[,-c(1:ncol(S))], model.matrix(~-1+I(S^2)*S)))
  if (nrow(S)==1) return(cbind(1, t(model.matrix(~-1+.^3, data=as.data.frame(S))[,-c(1:ncol(S))]), model.matrix(~-1+I(S^2)*S)))
}
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
norm_vec = function(x) sqrt(sum(x^2))
LSPISolve = function(simData, tolerance=1e-8, maxiter=50, degree=1, 
                     gamma=0.9, lambda=0, seed=NULL) {
  varList = list(pi=NULL, degree=NULL, transform=NULL, QFun=NULL, w=NULL, eps=NULL, criterion=NULL, iter=NULL)
  varList$degree = degree
  varList$transform = function(S) arbPoly(S, degree=varList$degree)
  longData = getLongData(Phi = simData$S, A = simData$A, U = simData$U)
  t = longData$t; n = max(longData$i); m = nrow(longData); finalTime = is.na(longData$U)
  U = longData$U[!finalTime]; AC = longData$A[!finalTime]
  Phi = longData %>% select(-c(A,U,i,t)) %>% as.matrix() %>% varList$transform()
  PhiC = Phi[!finalTime,]; PhiN = Phi[t>1,]; rm(Phi)
  terminal = rep(0, nrow(longData)) 
  terminalC = terminal[!finalTime]; terminalN = terminal[t>1]
  X = cbind(PhiC, AC*PhiC)
  scales = apply(X[,-c(1)], 2, sd)
  P = diag(c(0, scales^2))
  P[1,1] = P[ncol(X)/2+1,ncol(X)/2+1] = 0  
  XtDiffCN = crossprod(X, PhiC-(1-terminalN)*gamma*PhiN)
  XtXS = crossprod(X, X[,(ncol(X)/2+1):ncol(X)]) 
  
  iter=1
  eps=rep(1, ncol(X))
  if (!is.null(seed)) set.seed(seed)
  piN = rbinom(nrow(PhiN), 1, 0.5)
  wOld = w = rep(Inf, ncol(X)) 
  while (norm_vec(eps)>tolerance & iter<=maxiter) {
    Amat = cbind(XtDiffCN, XtXS-crossprod(X, (1-terminalN)*gamma*piN*PhiN))/m + lambda*P
    b = crossprod(X, U/m)
    w = solve(Amat)%*%b
    eps = norm_vec(w-wOld)
    piN = as.numeric(PhiN%*%w[(ncol(PhiN)+1):length(w)]>0)
    wOld = w
    iter = iter+1
  }
  varList$eps = eps
  varList$w = w
  varList$iter = iter
  
  varList$QFun = function(S, A, debug=F) {
    Phi = varList$transform(S)
    if (debug) browser()
    cbind(Phi, A*Phi)%*%varList$w
  }
  
  varList$pi = function(S) {
    Phi = varList$transform(S)
    as.numeric(Phi%*%varList$w[(ncol(Phi)+1):length(varList$w)]>0)
  }
  
  piN = as.numeric(PhiN%*%w[(ncol(PhiN)+1):length(w)]>0)
  varList$criterion = max(abs(t(X)%*%(U+(1-terminalN)*gamma*cbind(PhiN, piN*PhiN)%*%w-X%*%w)-m*lambda*P%*%w)) 
  return(varList)
}



#tree-based fitted Q-iteration fitting function
QComputationsTrees = function(simData, gamma=0.9, mtryArg=NULL, nodesizeArg=50, seed=42,
                              maxiter=200, ntreeArg=50, ...) {
  longData = getLongData(Phi = simData$S, A = simData$A, U = simData$U)
  t = longData$t; n = max(longData$i)
  finalTime = is.na(longData$U); U = longData$U[!finalTime]
  sLong = longData %>% select(-c(A,U,i,t)) %>% as.matrix()
  SC = sLong[!finalTime,]; SN = sLong[t>1,]; AC = as.factor(longData$A[!finalTime])
  if (is.null(mtryArg)) mtryArg=ncol(SC)+1
  actionSpace = sort(unique(AC))
  terminal = rep(0, nrow(longData))
  terminalC = terminal[!finalTime]; terminalN = terminal[t>1]
  
  QArrowOld = QArrow = SNList = fit = list()
  for (i in 1:length(actionSpace)) {
    QArrowOld[[i]] = rep(-Inf, length(U))
    QArrow[[i]] = rep(0, length(U))
  }
  iter = 1; eps = -1; asv = NULL
  while (iter<maxiter) {
    if (iter>1) {
    for (i in 1:length(actionSpace)) QArrow[[i]] = (1-terminalN)*predict(fit[[i]], SN, seed=seed)$predictions
    }
    QMax = rep(-Inf, length(U))
    for (i in 1:length(actionSpace)) QMax = pmax(QMax, QArrow[[i]])
    Bellman = U + gamma*QMax
      for (i in 1:length(actionSpace)) fit[[i]] = ranger(Bellman~., data=as.data.frame(cbind(SC,Bellman))[AC==actionSpace[i],], 
                                                         num.trees=ntreeArg, mtry=mtryArg, min.node.size=nodesizeArg, 
                                                         seed=seed, ...)
    if (iter>1) eps = mean(unlist(QArrow))-mean(unlist(QArrowOld))
    QArrowOld = QArrow
    iter = iter + 1
  } 
  
  QFun = function(S,A) {
    predictions = vector("double", nrow(S))
    colnames(S) = colnames(SC)
    for (i in 1:length(actionSpace)) {
      if (nrow(S)>1 & sum(A==actionSpace[i])>0) {
        predictions[A==actionSpace[i]] =  predict(fit[[i]], S[A==actionSpace[i],], seed=seed)$predictions
      }
      if (nrow(S)==1 & sum(A==actionSpace[i])>0) {
        predictions[A==actionSpace[i]] = predict(fit[[i]], S, seed=seed)$predictions
      }
    }
    return(predictions)
  }
  
  pi = function(S) {
    QPreds = matrix(ncol=length(actionSpace), nrow=nrow(S))
    for (i in 1:length(actionSpace)) QPreds[,i] = QFun(S, actionSpace[i])
    return(actionSpace[max.col(QPreds)])
  } 
  return(list(QFun=QFun, pi=pi, eps=eps))
}


#generate healthcare simulation
fac2dbl = function(x) as.double(as.character(x))
genmodH = function(n, capT, burn = 50, policyFun=NULL) {
  capT = capT + 1
  S = array(NA, dim = c(n, 2, capT))
  A = U = matrix(NA, nrow = n, ncol = capT)
  Sold = Snew = cbind(rnorm(n), rnorm(n))
  if (is.null(policyFun)) policyFun = function(x) rbinom(nrow(x), 1, 0.5)

  #burn-in to force stationarity upon initial states 
  for (j in 1:burn) {
    A0 = fac2dbl(policyFun(Sold))
    Snew[,1] = pmax(pmin(0.75*(2*A0 - 1) * Sold[,1] + 0.25*Sold[,1]*Sold[,2] + rnorm(n, 0, 0.25), 2), -2)
    Snew[,2] = pmax(pmin(0.75 *(1 - 2*A0) * Sold[,2] + 0.25*Sold[,1]*Sold[,2] + rnorm(n, 0, 0.25), 2), -2)
    Sold = Snew
  }
  S[,,1] = Snew
  
  for (t in 2:capT) {
    A[,t-1] = fac2dbl(policyFun(S[,,t-1]))
    S[,1,t] = pmax(pmin(0.75 * (2 * A[,t-1] - 1) * S[,1,t-1] + 0.25*S[,1,t-1]*S[,2,t-1] + rnorm(n, 0, 0.25), 2), -2)
    S[,2,t] = pmax(pmin(0.75 * (1 - 2 * A[,t-1]) * S[,2,t-1] + 0.25*S[,1,t-1]*S[,2,t-1] + rnorm(n, 0, 0.25), 2), -2)
    U[,t-1] = 2*S[,1,t]+S[,2,t]-0.25*(2*A[,t-1]-1)
  }
  return(list(S = S, A = A, U = U))
} 


#estimate policy by minimizing EMSBE
minEMSBE = function(S, A, terminalN=0, gamma=0.9, transform=linearT) {
    returnList = list(transform=NULL, beta=NULL, QFun=NULL, pi=NULL)
    returnList$transform = transform
    Phi = returnList$transform(S)[,-c(1)]
    X = model.matrix(~Phi+A*Phi)
    beta0 = rep(0, ncol(X))
    
    QFun = function(beta,S,A) {
        Phi = returnList$transform(S)[,-c(1)]
        if (length(A)==1) A=rep(A, nrow(Phi))
        X = model.matrix(~Phi+A*Phi)
        return(X %*% beta)
    }
    
    pi = function(beta,S) {
        Phi = returnList$transform(S)[,-c(1)]
        X = model.matrix(~Phi+rep(1,nrow(Phi))*Phi)
        selections = as.double(X[,-c(1:ncol(X)/2)] %*% beta[-c(1:ncol(X)/2)] > 0)
        return(selections)
    }
    
    loss = function(beta,S,A) mean((U+(1-terminalN)*gamma*pmax(QFun(beta,SN,0), QFun(beta,SN,1)) - QFun(beta,S,A))^2)
    returnList$beta = optim(par = beta0, fn = loss, S=S, A=A, method = 'BFGS')$par
    returnList$QFun = function(S,A) QFun(returnList$beta,S,A)
    returnList$pi = function(S) pi(returnList$beta,S)
    return(returnList)
}


#apply FQE using LSPI
LSPIEval = function(simData, policyFun, degree=1, lambda=0, gamma=0.9) {
  varList = list(pi=NULL, degree=NULL, transform=NULL, QFun=NULL, w=NULL, criterion=NULL)
  varList$pi = policyFun
  varList$degree = degree
  varList$transform = function(S) arbPoly(S, degree=varList$degree)
  longData = getLongData(Phi = simData$S, A = simData$A, U = simData$U)
  t = longData$t; n = max(longData$i); m = nrow(longData); finalTime = is.na(longData$U)
  U = longData$U[!finalTime]; AC = longData$A[!finalTime]
  S = longData %>% select(-c(A,U,i,t)) %>% as.matrix()
  Phi =  varList$transform(S)
  SC = S[!finalTime,]; SN = S[t>1,]; rm(S)
  PhiC = Phi[!finalTime,]; PhiN = Phi[t>1,]; rm(Phi)
  terminal = rep(0, nrow(longData))
  terminalC = terminal[!finalTime]; terminalN = terminal[t>1]
  X = cbind(PhiC, AC*PhiC)
  scales = apply(X[,-c(1)], 2, sd)
  P = diag(c(0, scales^2))
  P[1,1] = P[ncol(X)/2+1,ncol(X)/2+1] = 0  
  
  XtDiffCN = crossprod(X, PhiC-(1-terminalN)*gamma*PhiN)
  XtXS = crossprod(X, X[,(ncol(X)/2+1):ncol(X)]) 
  piN = fac2dbl(varList$pi(SN))
  Amat = cbind(XtDiffCN, XtXS-crossprod(X, (1-terminalN)*gamma*piN*PhiN))/m + lambda*P
  b = crossprod(X, U/m)
  w = solve(Amat)%*%b
  varList$w = w
  
  varList$QFun = function(S, A) {
    Phi = varList$transform(S)
    cbind(Phi, A*Phi)%*%varList$w
  }
  
  varList$criterion = max(abs(t(X)%*%(U+(1-terminalN)*gamma*cbind(PhiN, piN*PhiN)%*%w-X%*%w)-m*lambda*P%*%w)) 
  return(varList)
}


#Apply FQE using Tree-Based FQI
FQETreesSep = function(simData, policyFun, seed=42, mtryArg=2, nodesizeArg=1, gamma=0.9, maxiter=200, ntreeArg=50) {
  returnList = list(pi, QFun=NULL, eps=NULL, fit=NULL, SC=NULL, actionSpace=NULL, seed=seed)
  longData = getLongData(Phi = simData$S, A = simData$A, U = simData$U)
  t = longData$t; n = max(longData$i)
  finalTime = is.na(longData$U); U = longData$U[!finalTime]
  sLong = longData %>% select(-c(A,U,i,t)) %>% as.matrix()
  SC = sLong[!finalTime,]; SN = sLong[t>1,]; AC = as.factor(longData$A[!finalTime])
  if (is.null(mtryArg)) mtryArg=ncol(SC)+1
  actionSpace = sort(unique(AC))
  terminal = rep(0, nrow(longData))
  terminalC = terminal[!finalTime]; terminalN = terminal[t>1]
  returnList$pi = policyFun
  selectN = returnList$pi(SN)
  
  iter = 1; eps = -1; fit = list()
  QN = QN_old = rep(0, nrow(SN))
  while (iter<maxiter) {
    if (iter>1) {
      for (i in 1:length(actionSpace)) {
        if (sum(selectN==actionSpace[i])>0) QN[selectN==actionSpace[i]] = predict(fit[[i]], SN[selectN==actionSpace[i],], seed=seed)$predictions
      }
    }
    Bellman = U + (1-terminalN)*gamma*QN
    for (i in 1:length(actionSpace)) fit[[i]] = ranger(Bellman~., data=as.data.frame(cbind(SC,Bellman))[AC==actionSpace[i],], 
                                                         num.trees=ntreeArg, mtry=mtryArg, min.node.size=nodesizeArg, 
                                                         seed=seed)
    if (iter>1) eps = mean(QN)-mean(QN_old)
    QN_old = QN
    iter = iter + 1
  } 
  
  returnList$fit = fit; returnList$SC = SC; returnList$actionSpace = actionSpace; returnList$seed = seed
  returnList$QFun = function(S,A) {
    predictions = vector("double", nrow(S))
    colnames(S) = colnames(returnList$SC)
    for (i in 1:length(returnList$actionSpace)) {
      if (nrow(S)>1 & sum(A==returnList$actionSpace[i])>0) {
        preds = S[A==returnList$actionSpace[i],]
        if (sum(A==returnList$actionSpace[i])==1) {
          preds = matrix(preds, ncol=ncol(returnList$SC))
          colnames(preds) = colnames(returnList$SC)
        }
        predictions[A==returnList$actionSpace[i]] =  predict(returnList$fit[[i]], preds, seed=returnList$seed)$predictions
      }
      if (nrow(S)==1 & sum(A==returnList$actionSpace[i])>0) {
        predictions[A==returnList$actionSpace[i]] = predict(returnList$fit[[i]], S, seed=returnList$seed)$predictions
      }
    }
    return(predictions)
  }
  
  return(returnList)
}
