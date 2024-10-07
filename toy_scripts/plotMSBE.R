library(dplyr)
library(ggplot2)
load('session_x=0.5.RData')
set.seed(44)
color = ifelse(rank(msbe)>4, ifelse(rank(msbe)>15, 'High', 'Moderate'), 'Low')
color = factor(color, levels=c('High', 'Moderate', 'Low'))
summary_dat = data.frame(stand_returns,color) %>% group_by(color) %>% summarize(min=min(stand_returns), max=max(stand_returns))
msbe_plot = ggplot(mapping=aes(color=color)) + 
  geom_errorbar(data=summary_dat, mapping=aes(x=color, ymin=min, ymax=max)) + 
  geom_jitter(mapping=aes(x=color, y=stand_returns), position = position_jitter(width =.2)) +
  scale_color_manual(values = c('red', 'gold3', 'green3')) +
  theme_minimal(base_size = 16.5) + 
  theme(legend.position='none', axis.text=element_text(size=16.5), axis.title=element_text(size=17.5), 
        panel.grid.minor = element_blank()) +
  ylab('Return') + 
  xlab('MSBE Category')
pdf(file="msbe_plot.pdf", width=6, height=4)
plot(msbe_plot)
dev.off()
