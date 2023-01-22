#setting options and importing libraries
options(max.print=50)
library(dplyr, warn.conflicts=F)
library(ranger)


#generate Bicycle simulation
genmodBike = function(n=100, capT=1000, policyFun=NULL, dense=F) {
  #create objects for storage
  S = array(NA, dim = c(n, 4, capT+1))
  U = A = D = matrix(NA, nrow = n, ncol = capT+1)
  
  #set hyperparameters
  S[,,1] = cbind(matrix(0, nrow=n, ncol=4))
  D[,1] = FALSE
  if (is.null(policyFun)) policyFun = function(x) sample(seq(1,9), size=nrow(x), replace=T)
  delta_t = 0.01; v = 10/3.6; g = 9.82; d_CM = 0.3; c = 0.66; h = 0.94; M_c = 15; M_d = 1.7
  M_p = 60; M = M_c+M_p; r = 0.34; sigma_dot = v/r; I_bc = (13/3)*M_c*h^2 + M_p*(h+d_CM)^2
  I_dc = M_d*r^2; I_dv = (3/2)*M_d*r^2; I_dl = (1/2)*M_d*r^2; l = 1.11; c_reward = 0.1
  fallenThres = (12/180)*pi
  
  
  for (t in 1:capT) {
    #for non-terminal states
    if (sum(!D[,t])>0) {
      #calculate relevant quantities from current time point
      if (sum(!D[,t])==1) A[!D[,t],t] = policyFun(t(as.matrix(S[!D[,t],,t])))
      if (sum(!D[,t])>1) A[!D[,t],t] = policyFun(S[!D[,t],,t])
      invr_ft = abs(sin(S[!D[,t],3,t]))/l; invr_bt = abs(tan(S[!D[,t],3,t]))/l
      invr_CMt = (1/sqrt((1-c)^2+(1/invr_bt)^2))*(S[!D[,t],3,t]!=0)
      d_t = rep(-0.02, sum(!D[,t])) + 0.02*(A[!D[,t],t]%%3==2) + 0.04*(A[!D[,t],t]%%3==0)
      T_t = rep(-2, sum(!D[,t])) + 2*(A[!D[,t],t]>3) + 2*(A[!D[,t],t]>6)
      w_t = runif(sum(!D[,t]), -0.02, 0.02)
      phi_t = S[!D[,t],1,t] + atan(d_t+w_t)/h
      smallProd = M_d*r*(invr_ft+invr_bt) + M*h*invr_CMt
      largerProd = I_dc*sigma_dot*S[!D[,t],4,t] + sign(S[!D[,t],3,t])*v^2*smallProd
      giantProd = M*h*g*sin(phi_t)-cos(phi_t)*largerProd
      rq = S[!D[,t],3,t] + delta_t*S[!D[,t],4,t]
      
      #simulate from next time point
      S[!D[,t],1,t+1] = S[!D[,t],1,t] + delta_t*S[!D[,t],2,t]
      S[!D[,t],2,t+1] = S[!D[,t],2,t] + delta_t*(1/I_bc)*giantProd
      S[!D[,t],3,t+1] = rq*(abs(rq)<=80*pi/180) + (sign(rq)*80*pi/180)*(abs(rq)>80*pi/180)
      S[!D[,t],4,t+1] = (S[!D[,t],4,t]+delta_t*(T_t-I_dv*sigma_dot*S[!D[,t],2,t])/I_dl)*(abs(rq)<=80*pi/180)
      fallen = abs(S[!D[,t],1,t+1]) > fallenThres
      U[!D[,t],t] = - 1*fallen
      if (dense) U[!D[,t],t] = U[!D[,t],t] - (abs(S[!D[,t],1,t+1])-fallenThres)/fallenThres*(!fallen)
      D[!D[,t],t+1] =  fallen
    }
    
    #deal with terminal states
    D[D[,t],t+1] = TRUE
  }
  
  return(list(S=S, D=D, A=A, U=U))
}

#evaluate average reward under policy
evalPolicy = function(policyFun, n = 1000, capT = 1000, gamma=0.99, genmod=genmodBike, ...) {
  onlineData = genmod(n=n, capT=capT, policyFun=policyFun, ...)
  rewards = onlineData$U
  avgSurv = mean(apply(rewards, 1, function(x) max(which(!is.na(x)))))
  avgReturn = mean(apply(rewards, 1, function(x) sum(gamma^(0:(length(x)-1))*x, na.rm=T)))
  sumReturn = mean(apply(rewards, 1, function(x) sum(x, na.rm=T)))
  return(list(avgSurv=avgSurv, avgReturn=avgReturn, sumReturn=sumReturn))
}




#get data from 3-dimmensional arrays into long data frame or matrix
getLongData = function(Phi, A, U, mu = function(x) 0.5, returnMatrix=T, makeIntercept=F) {
  n = dim(Phi)[1]; capT = dim(Phi)[3]-1; p = dim(Phi)[2]
  PhiMat = aperm(Phi, perm = c(3,1,2)) 
  dim(PhiMat) = c(n*(capT+1),p)
  UVec = as.vector(t(U)); AVec = as.vector(t(A))
  i = rep(1:n, each=capT+1); t = rep(1:(capT+1), n)
  if (makeIntercept) {
    longData = cbind(1, PhiMat, AVec, UVec, i, t)
    p=p+1
  } else {
    longData = cbind(PhiMat, AVec, UVec, i, t)
  }
  start = ifelse(makeIntercept, 2, 1)
  missing = apply(as.matrix(longData[,start:(ncol(PhiMat)+start-1)]), 1, function(x) all(is.na(x)))
  longData = longData[!missing, ]
  colnames(longData) = c(paste0("S",1:p),"A","U",'i','t')
  if (!returnMatrix) longData = as.data.frame(longData)
  return(longData)
}


#tree-based fitted Q-iteration fitting function
QComputationsTrees = function(simData, gamma=0.99, mtryArg=NULL, nodesizeArg=50, seed=42, nthreadArg=NULL,  
                              inclAct=F, maxiter=200, joint=T, ntreeArg=50, terminal='auto',  ...) {
  longData = getLongData(Phi = simData$S, A = simData$A, U = simData$U, makeIntercept = F, returnMatrix = F)
  t = longData$t; n = max(longData$i)
  finalTime = is.na(longData$U); U = longData$U[!finalTime]
  sLong = longData %>% select(-c(A,U,i,t)) %>% as.matrix()
  SC = sLong[!finalTime,]; SN = sLong[t>1,]; AC = as.factor(longData$A[!finalTime])
  if (is.null(mtryArg)) mtryArg=ncol(SC)+1
  actionSpace = sort(unique(AC))
  if (is.null(terminal)) terminal = rep(0, nrow(longData))
  if (length(terminal)==1) if (terminal=='auto') terminal = as.numeric(finalTime)
  terminalC = terminal[!finalTime]; terminalN = terminal[t>1]
  
  QArrowOld = QArrow = SNList = fit = list()
  for (i in 1:length(actionSpace)) {
    QArrowOld[[i]] = rep(-Inf, length(U))
    QArrow[[i]] = rep(0, length(U))
  }
  iter = 1; eps = -1; asv = NULL
  if (inclAct) asv='AC'
  while (iter<maxiter) {
    if (iter>1) {
      if (joint) {
        for (i in 1:length(actionSpace)) {
          SNList[[i]] = cbind(SN, actionSpace[i])
          colnames(SNList[[i]]) = c(colnames(SC), 'AC')
          QArrow[[i]] = (1-terminalN)*predict(fit, SNList[[i]], seed=seed, num.threads=nthreadArg)$predictions
        }
      } else {
        for (i in 1:length(actionSpace)) QArrow[[i]] = (1-terminalN)*predict(fit[[i]], SN, seed=seed, num.threads=nthreadArg)$predictions
      }
    }
    QMax = rep(-Inf, length(U))
    for (i in 1:length(actionSpace)) QMax = pmax(QMax, QArrow[[i]])
    Bellman = U + gamma*QMax
    if (joint) {
      fit = ranger(Bellman~., data=as.data.frame(cbind(SC,AC,Bellman)), respect.unordered.factors=T, num.trees=ntreeArg, 
                   mtry=mtryArg, min.node.size=nodesizeArg, seed=seed, num.threads=nthreadArg, always.split.variables=asv, ...)
    } else {
      for (i in 1:length(actionSpace)) fit[[i]] = ranger(Bellman~., data=as.data.frame(cbind(SC,Bellman))[AC==actionSpace[i],], 
                                                         num.trees=ntreeArg, mtry=mtryArg-1, min.node.size=nodesizeArg, 
                                                         seed=seed, num.threads=nthreadArg, ...)
    }
    if (iter>1) eps = mean(unlist(QArrow))-mean(unlist(QArrowOld))
    QArrowOld = QArrow
    iter = iter + 1
  } 
  
  QFun = function(S,A) {
    predictions = vector("double", nrow(S))
    if (joint) {
      A = factor(x = A, levels = levels(AC), labels = levels(AC))
      SExt = cbind(S,A)
      colnames(SExt) = c(colnames(SC), 'AC')
      predictions = predict(fit, SExt, seed=seed)$predictions
    } else {
      colnames(S) = colnames(SC)
      for (i in 1:length(actionSpace)) {
        if (nrow(S)>1 & sum(A==actionSpace[i])>0) {
          predictions[A==actionSpace[i]] =  predict(fit[[i]], S[A==actionSpace[i],], seed=seed)$predictions
        }
        if (nrow(S)==1 & sum(A==actionSpace[i])>0) {
          predictions[A==actionSpace[i]] = predict(fit[[i]], S, seed=seed)$predictions
        }
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


#estimate policy by EMSBE minimization
linearT = function(S) cbind(1,S)
quadratic = function(S) cbind(1, model.matrix(~-1+.^2, data=as.data.frame(S)), S^2)
cubic = function(S) {
  if (nrow(S)>1) return(cbind(1, model.matrix(~-1+.^3, data=as.data.frame(S))[,-c(1:ncol(S))], model.matrix(~-1+I(S^2)*S)))
  if (nrow(S)==1) return(cbind(1, t(model.matrix(~-1+.^3, data=as.data.frame(S))[,-c(1:ncol(S))]), model.matrix(~-1+I(S^2)*S)))
}
minEMSBE = function(S, A, terminalN=0, gamma=0.9, transform=linearT) {
    returnList = list(transform=NULL, actionSpace=NULL, beta=NULL, QFun=NULL, pi=NULL)
    returnList$transform = transform
    Phi = returnList$transform(S)[,-c(1)]
    returnList$actionSpace = sort(unique(A))
    A = factor(A, levels=returnList$actionSpace, labels=returnList$actionSpace)
    X = model.matrix(~Phi+A*Phi)
    beta0 = rep(0, ncol(X))

    QFun = function(beta,S,A) {
        Phi = returnList$transform(S)[,-c(1)]
        if (is.vector(Phi)) Phi=t(Phi)
        A = factor(A, levels=returnList$actionSpace, labels=returnList$actionSpace)
        if (length(A)==1) A=rep(A, nrow(Phi))
        X = model.matrix(~Phi+A*Phi)
        return(X %*% beta)
    }
    
    QFun_grad = function(beta,S,A) {
        Phi = returnList$transform(S)[,-c(1)]
        A = factor(A, levels=returnList$actionSpace, labels=returnList$actionSpace)
        if (length(A)==1) A=rep(A, nrow(Phi))
        X = model.matrix(~Phi+A*Phi)
        return(X)
    }

    pi = function(beta,S) {
        QPreds = matrix(ncol=length(returnList$actionSpace), nrow=nrow(S))
        for (i in 1:length(returnList$actionSpace)) QPreds[,i] = QFun(beta, S, returnList$actionSpace[i])
        return(returnList$actionSpace[max.col(QPreds)])
    }

    loss = function(beta,S,A) {
        QMax = rep(-Inf, nrow(S))
        for (i in 1:length(returnList$actionSpace)) QMax = pmax(QMax, QFun(beta,S,returnList$actionSpace[i]))
        sqrt(mean((U + (1-terminalN)*gamma*QMax - QFun(beta,S,A))^2))
    }
    
    loss_grad = function(beta,S,A) {
        QMax = rep(-Inf, nrow(S))
        QMax_grad = matrix(0, nrow=nrow(S), ncol=length(beta0))
        for (i in 1:length(returnList$actionSpace)) {
            QMax = pmax(QMax, QFun(beta,S,returnList$actionSpace[i]))
            QMax_grad = ifelse(array(QMax==QFun(beta,S,returnList$actionSpace[i]), dim=dim(QMax_grad)), 
                               QFun_grad(beta,S,returnList$actionSpace[i]),
                               QMax_grad)
        }
        grad1 = (1/2) * 1/sqrt(mean((U + (1-terminalN)*gamma*QMax - QFun(beta,S,A))^2))
        grad2 = as.vector(2*(U + (1-terminalN)*gamma*QMax - QFun(beta,S,A)))
        grad3 = ((1-terminalN)*gamma*QMax_grad - QFun_grad(beta,S,A))
        return(grad1*as.vector(unlist(colMeans(grad2*grad3))))
    }
    
    returnList$beta = optim(par = beta0, fn = loss, gr=loss_grad, S=S, A=A, method = 'BFGS', control=list(trace=0))$par
    returnList$QFun = function(S,A) QFun(returnList$beta,S,A)
    returnList$pi = function(S) pi(returnList$beta,S)
    return(returnList)
}


#tree-based FQE fitting function
FQETree = function(simData, policyFun, mtryArg=NULL, nodesizeArg=50, seed=42,  
                   maxiter=200, gamma=0.99, ntreeArg=50) {
  returnList = list(pi, QFun=NULL, eps=NULL, fit=NULL, SC=NULL, actionSpace=NULL, seed=seed)
  longData = getLongData(Phi = simData$S, A = simData$A, U = simData$U, makeIntercept = F, returnMatrix = F)
  t = longData$t; n = max(longData$i)
  finalTime = is.na(longData$U); U = longData$U[!finalTime]
  sLong = longData %>% select(-c(A,U,i,t)) %>% as.matrix()
  SC = sLong[!finalTime,]; SN = sLong[t>1,]; AC = as.factor(longData$A[!finalTime])
  if (is.null(mtryArg)) mtryArg=ncol(SC)+1
  actionSpace = sort(unique(AC))
  terminal = as.numeric(finalTime)
  terminalC = terminal[!finalTime]; terminalN = terminal[t>1]
  
  returnList$pi = policyFun
  selectN = returnList$pi(SN)
  iter = 1; eps = -1
  QN = QN_old = rep(0, nrow(SN))
  
  while (iter<maxiter) {
    if (iter>1) {
      for (i in 1:length(actionSpace)) {
        if (sum(selectN==actionSpace[i])>0) {
          SNExt = cbind(SN[selectN==actionSpace[i],], actionSpace[i])
          colnames(SNExt) = c(colnames(SC), 'AC')
          QN[selectN==actionSpace[i]] = predict(fit, SNExt, seed=seed)$predictions
	}
      }
    }
    Bellman = U + (1-terminalN)*gamma*QN
    fit = ranger(Bellman~., data=as.data.frame(cbind(SC,AC,Bellman)), respect.unordered.factors=T, num.trees=ntreeArg, 
                 mtry=mtryArg, min.node.size=nodesizeArg, seed=seed)
    if (iter>1) eps = mean(QN)-mean(QN_old)
    QN_old = QN
    iter = iter + 1
  }
  
  returnList$fit = fit; returnList$SC = SC; returnList$AC = AC; returnList$actionSpace = actionSpace; returnList$seed = seed
  returnList$QFun = function(S,A) {
    predictions = vector("double", nrow(S))
    A = factor(x = A, levels = levels(returnList$AC), labels = levels(returnList$AC))
    SExt = cbind(S,A)
    colnames(SExt) = c(colnames(returnList$SC), 'AC')
    predictions = predict(returnList$fit, SExt, seed=returnList$seed)$predictions
    return(predictions)
  }
  
  return(returnList)
}
