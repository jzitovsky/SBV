#setup
args = commandArgs(trailingOnly=TRUE)
x_choose = ifelse(is.na(args[1]), 0.5, as.double(args[1]))
source('RFunctions.R')
options("scipen"=6, "digits"=6)


#calculate various policies and evaluate average returns
set.seed(42); n=20; capT=25
trainData = genmodS(n = n, capT = capT, x=x_choose)
degrees = seq(1,7)
lambdas = c(1e-6, 1e-3, 0.01, 0.02, 0.1, 1, 3e+5)
numPolicies = 29
methods = vector('character', numPolicies)
policies = list()
returns = vector('double', numPolicies)

i=1
for (degree in degrees) {
  for (lambda in lambdas) {
    if (degree>3  & lambda %in% c(0.01, 1)) next
    if (degree!=3 & lambda %in% c(1e-3, 0.02)) next
    maxiter = ifelse(degree>=5 & lambda==1e-6, 50, 300)
    n = ifelse(degree<5, 10000, 2000)
    policies[[i]] = QCompsRidge(trainData, transform=arbPoly, 
                                degree=degree, lambda=lambda, maxiter=maxiter)
    methods[i] = paste('degree=',degree,', lambda=',lambda, sep='')
    returns[i] = evalBeta(custom=T, x=x_choose,
                          policyFun=policies[[i]]$pi)
    i = i+1
  }
}





#simulate large test dataset
set.seed(43)
n=6000; capT=100; gamma=0.9
testData = genmodS(n=n, capT=capT, x=x_choose)
ldTest = getLongData(Phi = testData$S, A = testData$A, U = testData$U, returnMatrix = F)
finalTime = is.na(ldTest$U)
states = ldTest %>% filter(!finalTime) %>% select(-c(A,U,i,t,propensities)) %>% as.matrix()
statesN = ldTest %>% filter(t>1) %>% select(-c(A,U,i,t,propensities)) %>% as.matrix()
t = ldTest %>% select(t) %>% unlist() %>% as.vector()
U = ldTest[!finalTime,6]
A = ldTest[!finalTime,5]


#estimate true MSBE on large test dataset
msbe = vector('double', numPolicies)
calc_msbe = function(QRes, k_parm=100) {
  y = U + gamma*pmax(QRes$QFun(statesN, 0), QRes$QFun(statesN, 1))
  statesPred = states
  statesPred[,1] = statesPred[,1]*2
  trainModel0 = knnreg(x=as.matrix(statesPred[1:2e+5,][A[1:2e+5]==0,]), y=y[1:2e+5][A[1:2e+5]==0], k=k_parm)
  trainModel1 = knnreg(x=as.matrix(statesPred[1:2e+5,][A[1:2e+5]==1,]), y=y[1:2e+5][A[1:2e+5]==1], k=k_parm)
  errors = c(predict(trainModel0, statesPred[2e+5:3e+5,][A[2e+5:3e+5]==0 & t[2e+5:3e+5]<=25,])-QRes$QFun(states[2e+5:3e+5,][A[2e+5:3e+5]==0 & t[2e+5:3e+5]<=25,],0), predict(trainModel1, statesPred[2e+5:3e+5,][A[2e+5:3e+5]==1 & t[2e+5:3e+5]<=25,])-QRes$QFun(states[2e+5:3e+5,][A[2e+5:3e+5]==1 & t[2e+5:3e+5]<=25,],1))
  mean(errors^2)
}

i=1
for (degree in degrees) {
  for (lambda in lambdas) {
    if (degree>3  & lambda %in% c(0.01, 1)) next
    if (degree!=3 & lambda %in% c(1e-3, 0.02)) next
    msbe[i] = calc_msbe(policies[[i]]) 
    i = i+1
  }
}


#estimate true optimal Q-function values on large online dataset
set.seed(45)
n=10000; capT=100; gamma=0.9
onlineData = genmodS(n=n, capT=capT, x=x_choose, custom=T, policyFun=function(x) rep(1,nrow(x)), randomInit=T)
y = apply(onlineData$U, 1, function(x) sum(gamma^(0:(length(x)-1)) * x, na.rm=T))
X = cbind(1,onlineData$S[,1,1], onlineData$A[,1])
QStar = lm(y~X-1)
#var(cbind(1,states[,1],A)%*%QStar$coefficients)


#estimate MSE on large test dataset
calc_acc = function(QRes) return(mean(cbind(1,states[2e+5:3e+5,][t[2e+5:3e+5]<=25,1],A[2e+5:3e+5][t[2e+5:3e+5]<=25])%*%QStar$coefficients-QRes$QFun(states[2e+5:3e+5,][t[2e+5:3e+5]<=25,], A[2e+5:3e+5][t[2e+5:3e+5]<=25]))^2)
pred_errs = vector('double', numPolicies)
i=1
for (degree in degrees) {
  for (lambda in lambdas) {
    if (degree>3  & lambda %in% c(0.01, 1)) next
    if (degree!=3 & lambda %in% c(1e-3, 0.02)) next
    pred_errs[i] = calc_acc(policies[[i]])
    i = i+1
  }
}





#simulate held-out validation set
set.seed(45); n=5; capT=25
valData = genmodS(n = n, capT = capT, x=x_choose)

#get long data for train and validate
ldTrain = getLongData(Phi = trainData$S, A = trainData$A, U = trainData$U, returnMatrix = F)
finalTime = is.na(ldTrain$U)
S = ldTrain %>% filter(!finalTime) %>% select(-c(A,U,i,t,propensities)) %>% as.matrix()
SN = ldTrain %>% filter(t>1) %>% select(-c(A,U,i,t,propensities)) %>% as.matrix()
A = ldTrain[!finalTime,5]
U = ldTrain[!finalTime,6]
ldVal = getLongData(Phi = valData$S, A = valData$A, U = valData$U, returnMatrix = F)
finalTime = is.na(ldVal$U)
SVal = ldVal %>% filter(!finalTime) %>% select(-c(A,U,i,t,propensities)) %>% as.matrix()
SNVal = ldVal %>% filter(t>1) %>% select(-c(A,U,i,t,propensities)) %>% as.matrix()
AVal = ldVal[!finalTime,5]
UVal = ldVal[!finalTime,6]
degrees_reg = c(1,2,3)
lambdas_reg = c(1e-6, 1e-2, 1e+2, 1e+6)


#calculate SBV estimates
calc_SBV = function(QRes) {
  y = U + gamma*pmax(QRes$QFun(SN, 0), QRes$QFun(SN, 1))
  yVal = UVal + gamma*pmax(QRes$QFun(SNVal, 0), QRes$QFun(SNVal, 1))
  best_d = 0
  best_lambda = 0
  mse_vec = c()
  hyperparm_list = list()
  m = nrow(S)
  index = 1
  for (degree in degrees_reg) {
    transform = function(x) arbPoly(x, degree=degree)[,-c(1)]
    X = model.matrix(~transform(S)*as.factor(A))
    XVal = model.matrix(~transform(SVal)*as.factor(AVal))
    P = diag(c(0, apply(X[,-c(1)], 2, sd)))
    Amat = crossprod(X)/m
    b = t(X)%*%y/m
    for (lambda in lambdas_reg) {
      hyperparm_list[[index]] = c(degree, lambda)
      index = index + 1
      trainModel =  solve(Amat+lambda*P)%*%b
      mse_vec = c(mse_vec, mean((yVal - XVal%*%trainModel)^2))
    }
  }
  best_hyperparms = hyperparm_list[[which(mse_vec==min(mse_vec))[1]]]
  best_degree = best_hyperparms[1]
  best_lambda = best_hyperparms[2]
  transform = function(x) arbPoly(x, degree=best_degree)[,-c(1)]
  X = model.matrix(~transform(S)*as.factor(A))
  P = diag(c(0, apply(X[,-c(1)], 2, sd)))
  Amat = crossprod(X)/m
  b = t(X)%*%y/m
  XVal = model.matrix(~transform(SVal)*as.factor(AVal))
  return(XVal%*%(solve(Amat+best_lambda*P)%*%b))
}

sbv = vector('double', numPolicies)
i=1
for (degree in degrees) {
  for (lambda in lambdas) {
    if (degree>3  & lambda %in% c(0.01, 1)) next
    if (degree!=3 & lambda %in% c(1e-3, 0.02)) next
    sbv[i] = mean((calc_SBV(policies[[i]])-policies[[i]]$QFun(SVal, AVal))^2)
    i = i+1
  }
}




#calculate EMSBE estimates
calc_emsbe = function(QRes) mean((UVal+ gamma*pmax(QRes$QFun(SNVal, 0), QRes$QFun(SNVal, 1)) - QRes$QFun(SVal, AVal))^2)
emsbe = vector('double', numPolicies)
i=1
for (degree in degrees) {
  for (lambda in lambdas) {
    if (degree>3  & lambda %in% c(0.01, 1)) next
    if (degree!=3 & lambda %in% c(1e-3, 0.02)) next
    emsbe[i] = calc_emsbe(policies[[i]])
    i = i+1
  }
}


#adding true optimal Q-function to mix
returns = c(returns, evalBeta(x=x_choose, custom=T, policyFun=function(x) rep(1,nrow(x))))
methods = c(methods, 'optimal')
msbe = c(msbe, 0)
pred_errs = c(pred_errs, 0)

QStarFun = function(states, A) {
    if (length(A)==1) A = rep(A, nrow(states))
    cbind(1,states[,1],A)%*%QStar$coefficients
}
piStar = list(QFun=QStarFun)
sbv = c(sbv, mean((calc_SBV(piStar)-piStar$QFun(SVal, AVal))^2))
emsbe = c(emsbe, calc_emsbe(piStar))
#calc_msbe(piStar) #sanity check - 0.0198


#generate table
emsbe = round(emsbe, 6)
sbv = round(sbv, 6)
msbe = round(msbe, 6)
pred_errs = round(pred_errs, 6)
returns = round(returns, 6)
stand_returns = (returns-min(returns))/(max(returns)-min(returns))
bellman_table = data.frame(methods, stand_returns, pred_errs, msbe, emsbe, sbv)
#print(bellman_table)
cor_table = cor(bellman_table[,-c(1)], method = 'spearman')
write.csv(cor_table, paste('cor_table_x=', x_choose, '.csv', sep=''))
save.image(paste('session_x=', x_choose, '.RData', sep=''))
