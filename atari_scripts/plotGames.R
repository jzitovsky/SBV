#libraries
library(ggplot2)


#function to construct game plots
plot_game = function(target_table, sbv_table, wis_table, env, config='ddqn_SArch', jitter=F, jitter2=F) {
  #neccesary objects for plots
  wis_table = subset(wis_table, run==config)
  sbv_table = subset(sbv_table, run==config)
  target_table = subset(target_table, run==config)
  iters = wis_table$iter/40
  returns = returns_no_stopping = wis_table$returns
  returns = (returns-min(returns))/(max(returns)-min(returns))
  best_iter = which(returns==max(returns))
  sbv_iter = which(sbv_table$SBV==min(sbv_table$SBV))
  emsbe_iter = which(target_table$EMSBE==min(target_table$EMSBE))
  wis_iter = which(wis_table$WIS==max(wis_table$WIS))
  iter_choices = c(sbv_iter, emsbe_iter, wis_iter)
  methods = c('SBV', 'EMSBE', 'WIS')
  points = data.frame(x=iters[iter_choices], y=returns[iter_choices], method=methods)
  points$method = factor(points$method, levels = c('SBV', 'EMSBE', 'WIS'))
  if (jitter) {
    points$x[1] = points$x[1] - 0.5
    points$x[2] = points$x[2] + 0.5
  }
  if (jitter2) {
    points$x[1] = points$x[1] - 0.5
    points$x[3] = points$x[3] + 0.5
  }
  
  #construct plot
  game_plot = ggplot() + 
    geom_line(mapping = aes(iters, returns), color='dark gray') + 
    geom_point(data=points, aes(x=x, y=y, color=method), size=3, alpha=0.75) + 
    geom_vline(data=points, aes(xintercept=x, color=method), linetype='dashed', key_glyph = "path") +
    geom_hline(yintercept=returns[length(returns)], linetype='twodash') + 
    xlab('Iteration') + 
    ylab('Return') + 
    ggtitle(env) +
    theme(plot.title = element_text(hjust = 0.5, size=20),
          legend.title=element_text(size=19), legend.text=element_text(size=19),
          axis.title=element_text(size=19), axis.text=element_text(size=15),
          panel.grid.minor = element_blank()) +
    scale_color_manual(values = c("#0072B2", "#E69F00", "#009E73")) +
    guides(color=guide_legend(title='Method'))
    
  return(game_plot)
}


#function to load data and construct plot
process_data = function(dir, env='', config='ddqn_SArch', jitter=F, jitter2=F) {
  target_table = read.csv(paste(dir, '/target_dict.csv', sep=''), header = T, row.names=1)
  sbv_table = read.csv(paste(dir, '/sbv_dict.csv', sep=''), header = T, row.names=1)
  wis_table = read.csv(paste(dir, '/wis_dict.csv', sep=''), header = T, row.names=1)
  plot_obj = plot_game(target_table, sbv_table, wis_table, env, config, jitter, jitter2)

}




#construct plots
theme_set(theme_minimal(base_size = 18))
asterix5_plot = process_data('Asterix5')+theme(axis.title.y=element_blank(), axis.title.x=element_blank())
asterix2_plot = process_data('Asterix2')+theme(axis.title.y=element_blank(), plot.title=element_blank(), axis.title.x=element_blank())
asterix3_plot = process_data('Asterix3')+theme(axis.title.y=element_blank(), plot.title=element_blank())
breakout1_plot = process_data('Breakout1', jitter=T)+theme(axis.title.y=element_blank(), axis.title.x=element_blank())
breakout2_plot = process_data('Breakout2')+theme(axis.title.y=element_blank(), plot.title=element_blank(), axis.title.x=element_blank())
breakout3_plot = process_data('Breakout3', jitter2=T)+theme(axis.title.y=element_blank(), plot.title=element_blank())
pong1_plot = process_data('Pong1', config='dqn', jitter=T) + theme(axis.title.x=element_blank())
pong2_plot = process_data('Pong2', config='dqn', jitter=T) + theme(plot.title=element_blank(), axis.title.x=element_blank())
pong4_plot = process_data('Pong4', config='dqn', jitter=T) + theme(plot.title=element_blank())
seaquest1_plot = process_data('Seaquest1')+theme(axis.title.y=element_blank(), axis.title.x=element_blank())
seaquest2_plot = process_data('Seaquest2')+theme(axis.title.y=element_blank(), plot.title=element_blank(), axis.title.x=element_blank())
seaquest4_plot = process_data('Seaquest4')+theme(axis.title.y=element_blank(), plot.title=element_blank())

game_plot = ggpubr::ggarrange(pong1_plot + ggtitle('Pong'), breakout1_plot + ggtitle('Breakout'), 
                              asterix5_plot + ggtitle('Asterix'), seaquest1_plot + ggtitle('Seaquest'), 
                              pong2_plot, breakout2_plot, asterix2_plot, seaquest2_plot, 
                              pong4_plot, breakout3_plot, asterix3_plot, seaquest4_plot, 
                              ncol=4, nrow=3, common.legend=TRUE, legend='bottom')
pdf(file="game_plot.pdf", width=13, height=10)
plot(game_plot)
dev.off()
