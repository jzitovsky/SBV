#preprocessing
library(dplyr)
library(ggplot2)
top3 = spearman = matrix(nrow=10, ncol=3)
colnames(top3) = colnames(spearman) = c('sbv', 'emsbe', 'wis')


#extracting info from Bike
for (i in 1:10) {
  metrics = read.csv(paste('bike_scripts/allMetrics_seed=', 41+i, '.csv', sep=''))
  best = metrics %>% arrange(-survVec) %>% head(3) %>% select(survVec) %>% unlist() %>% mean()
  worst = metrics %>% arrange(survVec) %>% head(3) %>% select(survVec) %>% unlist() %>% mean()
  standardize = function(x) (x-worst)/(best-worst)
  top3[i,'sbv'] = metrics %>% arrange(sbvVec) %>% head(3) %>% select(survVec) %>% unlist() %>% mean() %>% standardize()
  spearman[i,'sbv'] = cor(metrics$sbvVec, metrics$survVec, method='spearman')
  top3[i,'emsbe'] = metrics %>% arrange(emsbeVec) %>% head(3) %>% select(survVec) %>% unlist() %>% mean() %>% standardize()
  top3[i,'wis'] = 0.5
  spearman[i,'emsbe'] = cor(metrics$emsbeVec, metrics$survVec, method='spearman')
  spearman[i,'wis'] = 0
}

bike_top3_means = colMeans(top3)
bike_top3_sds = apply(top3, 2, sd)
bike_spearman_means = colMeans(spearman)
bike_spearman_sds = apply(spearman, 2, sd)


#extracting info from mHealth
for (i in 1:10) {
  metrics = read.csv(paste('mHealth_scripts/allMetrics_run_seed=', 41+i, '.csv', sep=''))
  best = metrics %>% arrange(-returns) %>% head(3) %>% select(returns) %>% unlist() %>% mean()
  worst = metrics %>% arrange(returns) %>% head(3) %>% select(returns) %>% unlist() %>% mean()
  standardize = function(x) (x-worst)/(best-worst)
  top3[i,'sbv'] = metrics %>% arrange(sbv) %>% head(3) %>% select(returns) %>% unlist() %>% mean() %>% standardize()
  spearman[i,'sbv'] = cor(metrics$sbv, metrics$returns, method='spearman')
   top3[i,'emsbe'] = metrics %>% arrange(emsbe_val) %>% head(3) %>% select(returns) %>% unlist() %>% mean() %>% standardize()
   spearman[i,'emsbe'] = cor(metrics$emsbe_val, metrics$returns, method='spearman')
   top3[i,'wis'] = metrics %>% arrange(-wis) %>% head(3) %>% select(returns) %>% unlist() %>% mean() %>% standardize()
   spearman[i,'wis'] = cor(metrics$wis, metrics$returns, method='spearman')
}

mHealth_top3_means = colMeans(top3)
mHealth_top3_sds = apply(top3, 2, sd)
mHealth_spearman_means = colMeans(spearman)
mHealth_spearman_sds = apply(spearman, 2, sd)


#aggregating and preprocessing info
means = c(mHealth_top3_means, mHealth_spearman_means, bike_top3_means, bike_spearman_means)
sds = c(mHealth_top3_sds, mHealth_spearman_sds, bike_top3_sds, bike_spearman_sds)
metric = rep(rep(c('value', 'spearman'), each=3), 2)
env = rep(c('mHealth', 'Bike'), each=6)
method = rep(c('SBV', 'EMSBE', 'WIS'), 4)
df = data.frame(env, metric, method, means=abs(means), sds)


#making top-3 value plot
value_plot = ggplot(subset(df, metric=='value'),
                    aes(x=env, y=means, fill=method)) + 
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) +
  geom_errorbar(aes(ymin=means-sds, ymax=means+sds), width=.2,
                position=position_dodge(.9)) + 
  theme_minimal(base_size = 16.5) +
  theme(axis.text=element_text(size=16.5), axis.title=element_text(size=17.5),
        legend.text=element_text(size=16.5), legend.title=element_text(size=17.5),
        panel.grid.minor = element_blank(), panel.grid.major.x = element_blank()) + 
  scale_fill_manual(values=c('#E69F00','#0072B2','#009E73')) + 
  ylab('Top-3 Policy Value') +
  xlab('Environment') + 
  guides(fill=guide_legend(title="Method", byrow = TRUE)) + 
  coord_cartesian(ylim=c(0.5,1))

pdf(file="value_plot.pdf", width=8, height=4)
plot(value_plot)
dev.off()


#making spearman plot
spearman_plot = ggplot(subset(df, metric=='spearman'),
                    aes(x=env, y=means, fill=method)) + 
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) +
  geom_errorbar(aes(ymin=means-sds, ymax=means+sds), width=.2,
                position=position_dodge(.9)) + 
  theme_minimal(base_size = 16.5) +
  theme(axis.text=element_text(size=16.5), axis.title=element_text(size=17.5),
        legend.text=element_text(size=16.5), legend.title=element_text(size=17.5),
        panel.grid.minor = element_blank(), panel.grid.major.x = element_blank()) + 
  scale_fill_manual(values=c('#E69F00','#0072B2','#009E73')) + 
  ylab('Spearman\'s Rho') +
  xlab('Environment') + 
  guides(fill=guide_legend(title="Method", byrow = TRUE)) 

pdf(file="spearman_plot.pdf", width=8, height=4)
plot(spearman_plot)
dev.off()
