#simulate train data
source("RFunctions.R")
args = commandArgs(trailingOnly=TRUE)
seed = ifelse(is.na(args[1]), 44, as.double(args[1]))
n=160; nVal=80
nthreads = NULL
run = paste('seed=', seed, sep='')
print(paste('starting run ', run, sep='')) 
capT=500; gamma=0.99; set.seed(seed)
trainData = genmodBike(n=n, capT=capT) 


#specify hyperparm configurations
QOutputs = list()
mtryVals = c(5, 3, 1)
mnsVals = c(625, 125, 25, 5, 1)
maxiterVals = c(20, 200)
hyperparms = expand.grid(mtryVals, mnsVals, maxiterVals)
colnames(hyperparms) = c('mtry', 'mns', 'maxiter')
parmVec = vector('character', nrow(hyperparms))
for (i in 1:nrow(hyperparms)) {
  parmVec[i] = paste('mtry=', hyperparms[i,1], ', mns=', hyperparms[i,2], 
                    ', maxiter=', hyperparms[i,3],  sep='')
}
parmVec = c(parmVec, 'zero function')



#estimate policies and bootstrap returns
survVec = vector('double', nrow(hyperparms))
returnVec = vector('double', nrow(hyperparms))
for (i in 1:nrow(hyperparms)) {
  QOutputs[[i]] = QComputationsTrees(trainData, gamma=gamma, nthreadArg=nthreads, seed=seed,
                                    mtryArg=hyperparms[i,1], nodesizeArg=hyperparms[i,2],
                                    maxiter=hyperparms[i,3])
  evalObj = evalPolicy(QOutputs[[i]]$pi, n=100)
  survVec[i] = evalObj$avgSurv
  returnVec[i] = evalObj$avgReturn
#  print(parmVec[i])
#  print(survVec[i])
}
evalObj0=evalPolicy(policyFun=NULL, n=100)
survVec = c(survVec, evalObj0$avgSurv)
returnVec = c(returnVec, evalObj0$sumReturn)


#simulate held-out validation set
set.seed(seed+3); capTVal=500
valData = genmodBike(n = nVal, capT = capTVal)


#get long data for train and validate
ldTrain = getLongData(Phi = trainData$S, A = trainData$A, U = trainData$U, returnMatrix = F)
finalTimeT = is.na(ldTrain$U)
S = ldTrain %>% filter(!finalTimeT) %>% select(-c(A,U,i,t)) %>% as.matrix()
SN = ldTrain %>% filter(t>1) %>% select(-c(A,U,i,t)) %>% as.matrix()
A = ldTrain$A[!finalTimeT]; U = ldTrain$U[!finalTimeT]
ldVal = getLongData(Phi = valData$S, A = valData$A, U = valData$U, returnMatrix = F)
ldVal$i = ldVal$i + max(ldTrain$i)
terminal = as.numeric(finalTimeT)
terminalC = terminal[!finalTimeT]; terminalN = terminal[ldTrain$t>1]
finalTimeV = is.na(ldVal$U)
SVal = ldVal %>% filter(!finalTimeV) %>% select(-c(A,U,i,t)) %>% as.matrix()
SNVal = ldVal %>% filter(t>1) %>% select(-c(A,U,i,t)) %>% as.matrix()
AVal = ldVal$A[!finalTimeV]; UVal = ldVal$U[!finalTimeV]
terminalVal = as.numeric(finalTimeV)
terminalCVal = terminalVal[!finalTimeV]; terminalNVal = terminalVal[ldVal$t>1]


#more pre-processing
X = cbind(S, factor(A,levels=seq(1,9)))
colnames(X)[length(colnames(X))] = 'A'
XVal = cbind(SVal, factor(AVal,levels=seq(1,9)))
colnames(XVal)[length(colnames(XVal))] = 'A'
mseVec = vector('double', nrow(hyperparms)/2)


#calculate SBV estimates
calc_sbv = function(QRes) {
  y = as.vector(U + gamma*(1-terminalN)*QRes$QFun(SN, QRes$pi(SN)))
  yVal = as.vector(UVal + gamma*(1-terminalNVal)*QRes$QFun(SNVal, QRes$pi(SNVal)))
  for (i in 1:(nrow(hyperparms)/2)) {
    mtry=hyperparms$mtry[i]; mns = hyperparms$mns[i]
    trainModel = ranger(y~., data=cbind(X,y), respect.unordered.factors=T, num.trees=200, 
                        seed=seed, num.threads=nthreads, oob.error=F,
                        mtry=mtry, min.node.size=mns)
    mseVec[i] = mean((yVal - predict(trainModel, XVal, seed=seed)$predictions)^2)
  }
  bestParms = hyperparms[which(mseVec==min(mseVec))[1],]
  finalModel = ranger(y~., data=cbind(X,y), respect.unordered.factors=T, num.trees=200, 
                      seed=seed, num.threads=nthreads, oob.error=F,
                      mtry=bestParms$mtry, min.node.size=bestParms$mns)
  sbv = mean((predict(finalModel, XVal, seed=seed)$predictions - QRes$QFun(SVal, AVal))^2)
  return(sbv)
}

sbvVec = vector('double', nrow(hyperparms))
for (i in 1:nrow(hyperparms)) sbvVec[i] = sqrt(calc_sbv(QOutputs[[i]]))
QOutput0 = list(QFun = function(S,A) rep(0,nrow(S)), pi = function(S) rep(0,nrow(S)))
sbvVec = c(sbvVec, sqrt(calc_sbv(QOutput0)))



#calculate EMSBE estimates
calc_emsbe = function(QRes) {
#  emsbe_train = (U + gamma*(1-terminalN)*QRes$QFun(SN, QRes$pi(SN)) - QRes$QFun(S, A))^2
  emsbe_val = (UVal + gamma*(1-terminalNVal)*QRes$QFun(SNVal, QRes$pi(SNVal)) - QRes$QFun(SVal, AVal))^2
#  mean(c(emsbe_train, emsbe_val))
mean(c(emsbe_val))
}

emsbeVec = vector('double', nrow(hyperparms))
for (i in 1:nrow(hyperparms)) emsbeVec[i] = sqrt(calc_emsbe(QOutputs[[i]]))
emsbeVec = c(emsbeVec, sqrt(calc_emsbe(QOutput0)))




#calculate WIS estimates
AFull = c(A, AVal)
SFull = rbind(S, SVal)
ldFull = rbind(ldTrain, ldVal)
finalTimeF = c(finalTimeT, finalTimeV)
UFull = c(U, UVal)
calc_wis = function(QRes) {
  imp_weights_num = as.numeric(AFull==QRes$pi(SFull))
  imp_weights_den = 1/9
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

wisVec = vector('double', nrow(hyperparms))
for (i in 1:nrow(hyperparms)) wisVec[i] = calc_wis(QOutputs[[i]])


zero_wis = data.frame(imp_weights = 1,
                              i = ldFull$i[!finalTimeF],
                              t = ldFull$t[!finalTimeF],
                              U = UFull) %>%
  group_by(i) %>%
  mutate(cum_weights = cumprod(imp_weights)) %>%
  summarize(weighted_return = sum(cum_weights*gamma^(t-1)*U),
            sum_weights = sum(cum_weights*gamma^(t-1))) %>%
  summarize(value = sum(weighted_return)/sum(sum_weights)) %>%
  as.numeric() 
wisVec = c(wisVec, zero_wis)


#add in EMSBE estimates
QRes = minEMSBE(S,A,gamma=gamma,terminalN=terminalN)
parmVec = c(parmVec, 'emsbe_linear')
evalObj	= evalPolicy(policyFun=QRes$pi, capT=1000, gamma=gamma)
survVec = c(survVec, evalObj$avgSurv)
returnVec = c(returnVec, evalObj$sumReturn)
emsbeVec = c(emsbeVec, sqrt(calc_emsbe(QRes)))
sbvVec = c(sbvVec, sqrt(calc_sbv(QRes)))
wisVec = c(wisVec, calc_wis(QRes))


QRes = minEMSBE(S,A,transform=quadratic,gamma=gamma,terminalN=terminalN)
parmVec = c(parmVec, 'emsbe_quad')
evalObj = evalPolicy(policyFun=QRes$pi, capT=1000, gamma=gamma)
survVec = c(survVec, evalObj$avgSurv)
returnVec = c(returnVec, evalObj$sumReturn)
emsbeVec = c(emsbeVec, sqrt(calc_emsbe(QRes)))
sbvVec = c(sbvVec, sqrt(calc_sbv(QRes)))
wisVec = c(wisVec, calc_wis(QRes))



#aggregate estimates and save workspace
allMetrics = cbind(parmVec, survVec, returnVec, sbvVec, emsbeVec, wisVec)
print('script completed')
write.csv(allMetrics, paste('allMetrics_', run, '.csv', sep=''))
save.image(file=paste('session_', run, '.RData', sep=''))
