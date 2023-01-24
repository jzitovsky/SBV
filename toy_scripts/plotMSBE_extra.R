#load and process data
library(dplyr)
library(ggplot2)
load('session_x=0.5.RData')
set.seed(44)
rank_msbe = rank(msbe, ties.method = 'first')
color = ifelse(rank(msbe)>4, ifelse(rank(msbe)>15, 'High', 'Moderate'), 'Low')
color = factor(color, levels=c('High', 'Moderate', 'Low'))
summary_dat = data.frame(stand_returns,rank_msbe, color) 
msbe_plot = ggplot(mapping=aes(color=color)) + 
  geom_point(data=summary_dat, mapping=aes(x=rank_msbe, y=stand_returns)) + 
  scale_color_manual(values = c("#0072B2", "#E69F00", "#009E73")) +
  theme_minimal(base_size = 16.5) +
  theme(axis.text=element_text(size=16.5), axis.title=element_text(size=17.5), 
        legend.text=element_text(size=16.5), legend.title=element_text(size=17.5),
        legend.position = "none", panel.grid.minor.y = element_blank()) +
  guides(color=guide_legend(title="MSBE Category")) +
  ylab('Returns') + 
  xlab('MSBE Ranks') + 
  scale_x_reverse()

msbe_plot =  ggpubr::ggarrange(msbe_plot, ncol=1, nrow=1, common.legend=TRUE, legend='bottom')
plot(msbe_plot)
pdf(file="msbe_extra.pdf", width=6, height=4)
plot(msbe_plot)
dev.off()
