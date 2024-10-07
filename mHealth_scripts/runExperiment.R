#calculate various policies and evaluate average returns
source("RFunctions.R")
args = commandArgs(trailingOnly=TRUE)
seed = ifelse(is.na(args[1]), 42, as.double(args[1]))
n = 24; nVal=6; capT=25
run = paste('run_seed=', seed, sep='')
print(paste('starting run', run))
gamma=0.9; set.seed(seed)
trainData = genmodH(n=n, capT=capT)
degrees = c(1, 2, 3)
lambdas = c(0, 1e-2, 1, 100)
mnss = c(1, 5, 25)
mtrys = c(1,2)
numPolicies = length(degrees)*length(lambdas) + length(mnss)*length(mtrys)
methods = vector('character', numPolicies)
policies = list()
returns = vector('double', numPolicies)
i=1
for (degree in degrees) {
  for (lambda in lambdas) {
    policies[[i]] = LSPISolve(trainData, degree=degree, lambda=lambda, 
                              gamma=gamma, seed=seed)
    methods[i] = paste('degree=',degree,', lambda=', lambda, sep='')
    returns[i] = evalPolicy(policyFun=policies[[i]]$pi, capT=100, 
                            gamma=gamma)$sumReturn/100
#    print(returns[i])
    i = i+1
  }
}
for (mtry in mtrys) {
  for (mns in mnss) {
    policies[[i]] = QComputationsTrees(trainData, nodesizeArg=mns, mtryArg=mtry,
                                       gamma=gamma, maxiter=100, seed=seed)
    methods[i] = paste('mtry=',mtry,', mns=', mns, sep='')
    returns[i] = evalPolicy(policyFun=policies[[i]]$pi, capT=100, 
                            gamma=gamma)$sumReturn/100
#    print(returns[i])
    i = i+1
  }
}




#simulate held-out validation set
set.seed(45)
valData = genmodH(n = nVal, capT=capT)

#get long data for train and validate
ldTrain = getLongData(Phi = trainData$S, A = trainData$A, U = trainData$U)
finalTime = is.na(ldTrain$U)
S = ldTrain %>% filter(!finalTime) %>% select(-c(A,U,i,t)) %>% as.matrix()
SN = ldTrain %>% filter(t>1) %>% select(-c(A,U,i,t)) %>% as.matrix()
A = ldTrain$A[!finalTime]; U = ldTrain$U[!finalTime]
terminal = terminalC = terminalN = 0 
ldVal = getLongData(Phi = valData$S, A = valData$A, U = valData$U)
finalTimeVal = is.na(ldVal$U)
SVal = ldVal %>% filter(!finalTimeVal) %>% select(-c(A,U,i,t)) %>% as.matrix()
SNVal = ldVal %>% filter(t>1) %>% select(-c(A,U,i,t)) %>% as.matrix()
AVal = ldVal$A[!finalTimeVal]; UVal = ldVal$U[!finalTimeVal]
terminalVal = terminalCVal = terminalNVal = 0


#calculate SBV estimates
calc_SBV = function(QRes) {
  degrees_reg = c(1, 2, 3)
  lambdas_reg = c(0, 1e-6, 1e-2, 1, 1e+2)
  y = U + (1-terminalN)*gamma*pmax(QRes$QFun(SN, 0), QRes$QFun(SN, 1))
  yVal = UVal + (1-terminalNVal)*gamma*pmax(QRes$QFun(SNVal, 0), QRes$QFun(SNVal, 1))
  best_d = 0; best_lambda = 0
  mse_vec = c(); hyperparm_list = list()
  m = nrow(S); index = 1
  for (degree in degrees_reg) {
    transform = function(x) arbPoly(x, degree=degree)[,-c(1)]
    X = model.matrix(~transform(S)*as.factor(A))
    XVal = model.matrix(~transform(SVal)*as.factor(AVal))
    P = diag(c(0, apply(X[,-c(1)], 2, sd)))
    Amat = crossprod(X)/m
    b = t(X)%*%y/m
    for (lambda in lambdas_reg) {
      hyperparm_list[[index]] = c(degree, lambda, 'poly')
      index = index + 1
      trainModel =  solve(Amat+lambda*P)%*%b
      mse_vec = c(mse_vec, mean((yVal - XVal%*%trainModel)^2))
    }
  }
  mtry_reg = c(1,2,3)
  mns_reg = c(1, 5, 25, 125)
  trainRanger = cbind(S,A,y)
  colnames(trainRanger)[ncol(trainRanger)] = 'y'
  valRanger = cbind(SVal, AVal, yVal)
  colnames(valRanger) = colnames(trainRanger)
  for (mtry in mtry_reg) {
    for (mns in mns_reg) {
      trainModel = ranger(y~., data=trainRanger, num.trees=50, mtry=mtry, min.node.size=mns)
      hyperparm_list[[index]] = c(mtry, mns, 'rf')
      index = index + 1
      mse_vec = c(mse_vec, mean((yVal - predict(trainModel, valRanger)$predictions)^2))
    }
  }
  
  best_hyperparms = hyperparm_list[[which(mse_vec==min(mse_vec))[1]]]
  if (best_hyperparms[3]=='poly') {
    best_degree = as.numeric(best_hyperparms[1])
    best_lambda = as.numeric(best_hyperparms[2])
    transform = function(x) arbPoly(x, degree=best_degree)[,-c(1)]
    X = model.matrix(~transform(S)*as.factor(A))
    P = diag(c(0, apply(X[,-c(1)], 2, sd)))
    Amat = crossprod(X)/m
    b = t(X)%*%y/m
    XVal = model.matrix(~transform(SVal)*as.factor(AVal))
    sbv_val = XVal%*%(solve(Amat+best_lambda*P)%*%b)
    return(sqrt(mean((sbv_val-QRes$QFun(SVal, AVal))^2)))
  }
  if (best_hyperparms[3]=='rf') {
    best_mtry = as.numeric(best_hyperparms[1])
    best_mns = as.numeric(best_hyperparms[2])
    trainModel = ranger(y~., data=trainRanger, num.trees=200, mtry=mtry, min.node.size=mns)
    sbv_val = predict(trainModel, valRanger)$predictions
    return(sqrt(mean((sbv_val-QRes$QFun(SVal, AVal))^2)))
  }
}

sbv = vector('double', numPolicies)
i=1
for (degree in degrees) {
  for (lambda in lambdas) {
    sbv[i] = calc_SBV(policies[[i]])
    i = i+1
  }
}
for (mtry in mtrys) {
  for (mns in mnss) {
    sbv[i] = calc_SBV(policies[[i]])
    i = i+1
  }
}


#calculate EMSBE estimates
calc_emsbe = function(QRes) {
  emsbe_val = (UVal+ (1-terminalNVal)*gamma*pmax(QRes$QFun(SNVal, 0), QRes$QFun(SNVal, 1)) - QRes$QFun(SVal, AVal))^2
  emsbe_train = (U+(1-terminalN)*gamma*pmax(QRes$QFun(SN, 0), QRes$QFun(SN, 1)) - QRes$QFun(S, A))^2
  emsbe = sqrt(mean(c(emsbe_val, emsbe_train)))
  emsbe_val = sqrt(mean(emsbe_val))
  return(list(emsbe=emsbe, emsbe_val=emsbe_val))
}
emsbe = vector('double', numPolicies)
emsbe_val = vector('double', numPolicies)
i=1
for (degree in degrees) {
  for (lambda in lambdas) {
    emsbeObj = calc_emsbe(policies[[i]])
    emsbe[i] = emsbeObj$emsbe
    emsbe_val[i] = emsbeObj$emsbe_val 
    i = i+1
  }
}
for (mtry in mtrys) {
  for (mns in mnss) {
    emsbeObj = calc_emsbe(policies[[i]])
    emsbe[i] = emsbeObj$emsbe           
    emsbe_val[i] = emsbeObj$emsbe_val
    i = i+1
  }
}


#calculate WIS estimates
AFull = c(A, AVal)
SFull = rbind(S, SVal)
ldFull = rbind(ldTrain, ldVal)
finalTimeF = c(finalTime, finalTimeVal)
UFull = c(U, UVal)
calc_wis = function(QRes) {
  imp_weights_num = as.numeric(AFull==QRes$pi(SFull))
  imp_weights_den = 1/2
  imp_weights_long = data.frame(imp_weights = imp_weights_num/imp_weights_den,
                                i = ldFull$i[!finalTimeF],
                                t = ldFull$t[!finalTimeF],
                                U = UFull)

  imp_weights_long  = imp_weights_long %>%
    group_by(i) %>%
    mutate(cum_weights = cumprod(imp_weights))

  value = imp_weights_long %>%
    group_by(i) %>%
    summarize(weighted_return = sum(cum_weights*gamma^(t-1)*U),
              sum_weights = sum(cum_weights*gamma^(t-1))) %>%
    summarize(value = sum(weighted_return)/sum(sum_weights)) %>%
    as.numeric()

  return(value)
}

wis = vector('double', numPolicies)
i=1
for (degree in degrees) {
  for (lambda in lambdas) {
    wis[i] = calc_wis(policies[[i]])
    i = i+1
  }
}
for (mtry in mtrys) {
  for (mns in mnss) {
    wis[i] = calc_wis(policies[[i]])
    i = i+1
  }
}



#add minimum EMSBE policies
QRes = minEMSBE(S,A)
methods = c(methods, 'emsbe_degree=1')
returns = c(returns, evalPolicy(policyFun=QRes$pi, capT=100, gamma=gamma)$sumReturn/100)
sbv = c(sbv, sqrt(calc_SBV(QRes)))
emsbeObj = calc_emsbe(QRes)
emsbe = c(emsbe, emsbeObj$emsbe)
emsbe_val = c(emsbe_val, emsbeObj$emsbe_val)
wis = c(wis, calc_wis(QRes))

QRes = minEMSBE(S,A,transform=quadratic)
methods = c(methods, 'emsbe_degree=2')
returns = c(returns, evalPolicy(policyFun=QRes$pi, capT=100, gamma=gamma)$sumReturn/100)
sbv = c(sbv, sqrt(calc_SBV(QRes)))
emsbeObj = calc_emsbe(QRes)
emsbe = c(emsbe, emsbeObj$emsbe)
emsbe_val = c(emsbe_val, emsbeObj$emsbe_val)
wis = c(wis, calc_wis(QRes))

QRes = minEMSBE(S,A,transform=cubic)
methods = c(methods, 'emsbe_degree=3')
returns = c(returns, evalPolicy(policyFun=QRes$pi, capT=100, gamma=gamma)$sumReturn/100)
sbv = c(sbv, sqrt(calc_SBV(QRes)))
emsbeObj = calc_emsbe(QRes)
emsbe = c(emsbe, emsbeObj$emsbe)
emsbe_val = c(emsbe_val, emsbeObj$emsbe_val)
wis = c(wis, calc_wis(QRes))


#generate table
emsbe = round(emsbe, 6)
emsbe_val = round(emsbe_val, 6)
sbv = round(sbv, 6)
returns = round(returns, 6)
wis = round(wis, 6)
bellman_table = data.frame(methods, returns, emsbe, emsbe_val, sbv, wis)
#print(bellman_table)
allMetrics = bellman_table
write.csv(allMetrics, paste('allMetrics_', run, '.csv', sep=''))
save.image(file=paste('session_', run, '.RData', sep=''))
print(paste('ending run', run))
