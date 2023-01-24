#libraries
library(ggplot2)

#load and process data
k=seq(1,5)
games = c('Pong', 'Breakout', 'Asterix', 'Seaquest')
emsbe = sbv = wis = c()

plot_game = function(game) {
  for (k_vals in k) {
    topk_table = read.csv(paste('top', k_vals, '_dict_oracle.csv', sep=''), header=T, row.names=1)
    emsbe = c(emsbe, topk_table[game, 'EMSBE'])
    sbv = c(sbv, topk_table[game, 'SBV'])
    wis = c(wis, topk_table[game, 'WIS'])
  }
  dat = data.frame(k = rep(k,3), topk = c(emsbe,sbv,wis), 
                   estimate=c(rep('EMSBE',5), rep('SBV',5), rep('WIS',5)))
  
  game_plot = ggplot(data=dat, aes(x=k, y=topk, color=estimate)) +
    geom_line() +
    geom_point() + 
    scale_color_manual(values = c("#E69F00", "#0072B2", "#009E73")) +
    theme_minimal() + 
    ylab('Top-k Return') +
    ggtitle(game) +
    guides(color=guide_legend(title='Method')) + 
    theme(plot.title = element_text(hjust = 0.5, size=20),
          legend.title=element_text(size=19), legend.text=element_text(size=19),
          axis.title=element_text(size=19), axis.text=element_text(size=15),
          panel.grid.minor = element_blank())
  return(game_plot)
}


theme_set(theme_minimal(base_size = 18))
pong_plot = plot_game('Pong')
breakout_plot = plot_game('Breakout') + theme(axis.title.y=element_blank())
asterix_plot = plot_game('Asterix') + theme(axis.title.y=element_blank())
seaquest_plot = plot_game('Seaquest') + theme(axis.title.y=element_blank())
game_plot = ggpubr::ggarrange(pong_plot, breakout_plot, asterix_plot, seaquest_plot,
                              nrow=1, common.legend=TRUE, legend='bottom')
pdf(file="atari_oracle.pdf", width=13, height=4.5)
plot(game_plot)
dev.off()
