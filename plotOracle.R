library(dplyr)
library(ggplot2)
k_vals = seq(1,5)
super_topk = matrix(nrow=length(k_vals), ncol=3)
topk = matrix(nrow=10, ncol=3)
colnames(topk) = colnames(super_topk) = c('sbv', 'emsbe', 'wis')
for (k in k_vals) {
  for (i in 1:10) {
    metrics = read.csv(paste('mHealth_scripts/allMetrics_run_seed=', 41+i, '.csv', sep=''))
    best = metrics %>% arrange(-returns) %>% head(1) %>% select(returns) %>% unlist()
    worst = metrics %>% arrange(returns) %>% head(1) %>% select(returns) %>% unlist() 
    standardize = function(x) (x-worst)/(best-worst)
    topk[i,'sbv'] = metrics %>% arrange(sbv) %>% head(k) %>% select(returns) %>% unlist() %>% max() %>% standardize()
    topk[i,'emsbe'] = metrics %>% arrange(emsbe_val) %>% head(k) %>% select(returns) %>% unlist() %>% max() %>% standardize()
    topk[i,'wis'] = metrics %>% arrange(-wis) %>% head(k) %>% select(returns) %>% unlist() %>% max() %>% standardize()
  }
  super_topk[k,] = colMeans(topk)
}

dat = data.frame(k = rep(k_vals,3), topk = super_topk %>% unlist() %>% as.vector(), 
                   estimate=c(rep('SBV',5), rep('EMSBE',5), rep('WIS',5)))
mHealth_plot = ggplot(data=dat, aes(x=k, y=topk, color=estimate)) +
    geom_line() +
    geom_point() + 
    scale_color_manual(values = c("#E69F00", "#0072B2", "#009E73")) +
    theme_minimal() + 
    ylab('Top-k Policy Value') +
    ggtitle('mHealth') +
    guides(color=guide_legend(title='Method')) + 
    theme_minimal(base_size = 16.5) + 
    theme(plot.title = element_text(hjust = 0.5, size=18),
          legend.title=element_text(size=17.5), legend.text=element_text(size=16.5),
          axis.title=element_text(size=17.5), axis.text=element_text(size=16.5), axis.title.y=element_blank(),
          panel.grid.minor = element_blank()) 





for (k in k_vals) {
  for (i in 1:10) {
    metrics = read.csv(paste('bike_scripts/allMetrics_seed=', 41+i, '.csv', sep=''))
    best = metrics %>% arrange(-survVec) %>% head(1) %>% select(survVec) %>% unlist()
    worst = metrics %>% arrange(survVec) %>% head(1) %>% select(survVec) %>% unlist() 
    standardize = function(x) (x-worst)/(best-worst)
    topk[i,'sbv'] = metrics %>% arrange(sbvVec) %>% head(k) %>% select(survVec) %>% unlist() %>% max() %>% standardize()
    topk[i,'emsbe'] = metrics %>% arrange(emsbeVec) %>% head(k) %>% select(survVec) %>% unlist() %>% max() %>% standardize()
    topk[i,'wis'] = 0
  }
  super_topk[k,] = colMeans(topk)
}

dat = data.frame(k = rep(k_vals,3), topk = super_topk %>% unlist() %>% as.vector(), 
                   estimate=c(rep('SBV',5), rep('EMSBE',5), rep('WIS',5)))
bike_plot = ggplot(data=dat, aes(x=k, y=topk, color=estimate)) +
    geom_line() +
    geom_point() + 
    scale_color_manual(values = c("#E69F00", "#0072B2", "#009E73")) +
    theme_minimal() + 
    ylab('Top-k Policy Value') +
    ggtitle('Bicycle') +
    guides(color=guide_legend(title='Method')) + 
    theme(plot.title = element_text(hjust = 0.5, size=18),
          legend.title=element_text(size=17.5), legend.text=element_text(size=16.5),
          axis.title=element_text(size=17.5), axis.text=element_text(size=16.5),
          panel.grid.minor = element_blank()) 

small_plot = ggpubr::ggarrange(bike_plot, mHealth_plot,
                              nrow=1, common.legend=TRUE, legend='bottom')
pdf(file="oracle_plot.pdf", width=6, height=4.5)
plot(small_plot)
dev.off()
