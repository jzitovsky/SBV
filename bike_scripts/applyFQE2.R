#calculate various policies and evaluate average returns
args = commandArgs(trailingOnly=TRUE)
seed = 42
load(paste('session_seed=', seed, '.RData', sep=''))
source('RFunctions.R')
fullData = vector('list', length(trainData))
names(fullData) = names(trainData)
fullData$S = abind::abind(trainData$S, valData$S, along=1)
fullData$A = abind::abind(trainData$A, valData$A, along=1)
fullData$U = abind::abind(trainData$U, valData$U, along=1)
fullData$D = abind::abind(trainData$D, valData$D, along=1)
SInitF = fullData$S[,,1]
fqeVec = vector('double', length(parmVec))
EvalFunc = function(pi) FQETree(simData=fullData, policyFun=pi, seed=seed, 
                                 nodesizeArg=5, mtryArg=3, maxiter=200)
i=1
EvalObj = EvalFunc(QOutputs[[i]]$pi)
fqeVec[i] = mean(EvalObj$QFun(SInitF, EvalObj$pi(SInitF)))


for (i in 1:nrow(hyperparms)) {
  EvalObj = EvalFunc(QOutputs[[i]]$pi)
  fqeVec[i] = mean(EvalObj$QFun(SInitF, EvalObj$pi(SInitF)))
#  print(parmVec[i])
#  print(survVec[i])
#  print(fqeVec[i])
}

i=i+1
#print(parmVec[i])
fqeVec[i] = NA
i=i+1

#add minimum EMSBE policies
#print(parmVec[i])
QRes = minEMSBE(S,A)
EvalObj = EvalFunc(QRes$pi)
fqeVec[i] = mean(EvalObj$QFun(SInitF, EvalObj$pi(SInitF)))
i = i+1

#print(parmVec[i])
QRes = minEMSBE(S,A,transform=quadratic)
EvalObj = EvalFunc(QRes$pi)
fqeVec[i] = mean(EvalObj$QFun(SInitF, EvalObj$pi(SInitF)))
i = i+1


#generate table
fqeVec = round(fqeVec, 6)
bellman_table = data.frame(parmVec, survVec, sbvVec, emsbeVec, wisVec, fqeVec)
options(max.print=1000)
print(bellman_table)
write.csv(bellman_table, 'fqe2_table.csv')
