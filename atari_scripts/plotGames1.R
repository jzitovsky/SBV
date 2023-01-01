#libraries
library(ggplot2)


#function to construct game plots
plot_game = function(target_table, sbv_table, wis_table, env, config='ddqn_SArch', jitter=F) {
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
          legend.text = element_text(size=15)) +
    scale_color_manual(values = c("#0072B2", "#E69F00", "#009E73")) +
    guides(color=guide_legend(title='Method'))
 
    
  return(game_plot)
}



#construct plots
theme_set(theme_minimal(base_size = 15))
#args = commandArgs(trailingOnly=TRUE)
args = c('Asterix5', 'Breakout1', 'Pong1', 'Seaquest1')
target_table_asterix = read.csv(paste(args[1], '/target_dict.csv', sep=''), header = T, row.names=1)
sbv_table_asterix = read.csv(paste(args[1], '/sbv_dict.csv', sep=''), header = T, row.names=1)
wis_table_asterix = read.csv(paste(args[1], '/wis_dict.csv', sep=''), header = T, row.names=1)
asterix_plot = plot_game(target_table_asterix, sbv_table_asterix, wis_table_asterix, env='Asterix')

target_table_breakout = read.csv(paste(args[2], '/target_dict.csv', sep=''), header = T, row.names=1)
sbv_table_breakout = read.csv(paste(args[2], '/sbv_dict.csv', sep=''), header = T, row.names=1)
wis_table_breakout = read.csv(paste(args[2], '/wis_dict.csv', sep=''), header = T, row.names=1)
breakout_plot = plot_game(target_table_breakout, sbv_table_breakout, wis_table_breakout, env='Breakout', jitter=T)

target_table_pong = read.csv(paste(args[3], '/target_dict.csv', sep=''), header = T, row.names=1)
sbv_table_pong = read.csv(paste(args[3], '/sbv_dict.csv', sep=''), header = T, row.names=1)
wis_table_pong = read.csv(paste(args[3], '/wis_dict.csv', sep=''), header = T, row.names=1)
pong_plot = plot_game(target_table_pong, sbv_table_pong, wis_table_pong, env='Pong',  config='dqn', jitter=T)

target_table_seaquest = read.csv(paste(args[4], '/target_dict.csv', sep=''), header = T, row.names=1)
sbv_table_seaquest = read.csv(paste(args[4], '/sbv_dict.csv', sep=''), header = T, row.names=1)
wis_table_seaquest = read.csv(paste(args[4], '/wis_dict.csv', sep=''), header = T, row.names=1)
seaquest_plot = plot_game(target_table_seaquest, sbv_table_seaquest, wis_table_seaquest, env='Seaquest')


game_plot = ggpubr::ggarrange(pong_plot, breakout_plot, asterix_plot, seaquest_plot, 
                              ncol=4, nrow=1, common.legend=TRUE, legend='bottom')
pdf(file="game_plot1.pdf", width=13, height=4)
plot(game_plot)
dev.off()
