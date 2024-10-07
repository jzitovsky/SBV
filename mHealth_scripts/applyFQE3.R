#calculate various policies and evaluate average returns
args = commandArgs(trailingOnly=TRUE)
seed=42; method='rf'; lambda_eval=NA
load(paste('session_run_seed=', seed, '.RData', sep=''))
source('RFunctions.R')
fullData = vector('list', 3)
names(fullData) = names(trainData)
fullData$S = abind::abind(trainData$S, valData$S, along=1)
fullData$A = abind::abind(trainData$A, valData$A, along=1)
fullData$U = abind::abind(trainData$U, valData$U, along=1)
if (method=='lspi') {
  EvalFunc = function(pi) LSPIEval(simData=fullData, policyFun=pi, degree=2, lambda=lambda_eval)
} else if (method=='rf') {
  EvalFunc = function(pi) FQETreesSep(simData=fullData, policyFun=pi, seed=seed, maxiter=100)
}
print(paste('run with seed', seed, 'method', method, 'lambda', lambda_eval))


i=1
SInitF = fullData$S[,,1]
values = vector('double', length(methods))
for (degree in degrees) {
  for (lambda in lambdas) {
    EvalObj = EvalFunc(policies[[i]]$pi)
    values[i] = mean(EvalObj$QFun(SInitF, fac2dbl(EvalObj$pi(SInitF))))
    i = i+1
  }
}
for (mtry in mtrys) {
  for (mns in mnss) {
    EvalObj = EvalFunc(policies[[i]]$pi)
    values[i] = mean(EvalObj$QFun(SInitF, fac2dbl(EvalObj$pi(SInitF))))
    i = i+1
  }
}



#add minimum EMSBE policies
#print(methods[i])
QRes = minEMSBE(S,A)
EvalObj = EvalFunc(QRes$pi)
values[i] = mean(EvalObj$QFun(SInitF, fac2dbl(EvalObj$pi(SInitF))))
i = i+1

#print(methods[i])
QRes = minEMSBE(S,A,transform=quadratic)
EvalObj = EvalFunc(QRes$pi)
values[i] = mean(EvalObj$QFun(SInitF, fac2dbl(EvalObj$pi(SInitF))))
i = i+1

#print(methods[i])
QRes = minEMSBE(S,A,transform=cubic)
EvalObj = EvalFunc(QRes$pi)
values[i] = mean(EvalObj$QFun(SInitF, fac2dbl(EvalObj$pi(SInitF))))
i = i+1

#generate table
fqe = round(values, 6)
bellman_table = data.frame(methods, returns, emsbe, emsbe_val, sbv, wis, fqe)
options(max.print=1000)
print(select(bellman_table, -c(emsbe)))
write.csv(bellman_table, 'fqe3_table.csv')
