#libraries
library(ggplot2)

#load and process data
x=c(0.75,0.7,0.65,0.6,0.55,0.5)
emsbe_valid = sbv_valid = c()
for (x_vals in x) {
  cor_table = read.csv(paste('cor_table_x=', x_vals, '.csv', sep=''), header=T, row.names=1)
  emsbe_valid = c(emsbe_valid, cor_table['msbe', 'emsbe'])
  sbv_valid = c(sbv_valid, cor_table['msbe', 'sbv'])
}
dat = data.frame(x = rep(x,2), estimates = c(emsbe_valid,sbv_valid), 
                 estimate=c(rep('emsbe',6), rep('sbv',6)))

#make plot
noise_plot = ggplot(data=dat, aes(x=0.75-x, y=estimates, color=estimate)) +
  geom_line() +
  geom_point() + 
  ylab('Spearman Corr w/ MSBE') +
  xlab('stochasticity') +
  theme_minimal(base_size = 15) 
noise_plot = ggpubr::ggarrange(noise_plot, ncol=1, nrow=1, common.legend=TRUE, legend='bottom')
plot(noise_plot)

#save plot
pdf(file="noise_plot.pdf", width=6, height=4)
plot(noise_plot)
dev.off()
