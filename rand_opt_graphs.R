library(tidyverse)
library(ggpubr)

# Fitness by iteration
tsm = read_csv('travel_iters.csv')
om = read_csv('onemax_iters.csv')
fp = read_csv('fourpeaks_iters.csv')

fp %>% 
  filter(Iterations < 3000, Samples != 10) %>% 
  ggplot(aes(Iterations, Score, color = Algorithm))+
  geom_line(size = 1) +
  facet_wrap(~Samples, scales = 'free')+
  labs(title = 'Four Peaks Problem: Max Peak by Space Size')+
  theme_bw()

# Simulated annealing performed well on smaller sample spaces
# Genetic and Mimic eventualy reached the global maxima in a few thousand iterations
#
om %>% 
  filter(Samples != 10, Iterations < 200) %>% 
  ggplot(aes(Iterations, Score, color = Algorithm))+
  geom_line(size = 1) +
  facet_wrap(~Samples, scales = 'free')+
  labs(title = 'One Max Problem: Output Sum by Sequence Size (Iterations Restricted)')+
  theme_bw()
om %>% 
  filter(Samples != 10, Iterations < 2000) %>% 
  ggplot(aes(Iterations, Score, color = Algorithm))+
  geom_line(size = 1) +
  facet_wrap(~Samples, scales = 'free')+
  labs(title = 'One Max Problem: Output Sum by Sequence Size')+
  theme_bw()

tsm %>% 
  filter(Samples != 10) %>% 
  ggplot(aes(Iterations, Score, color = Algorithm))+
  geom_line(size = 1) +
  facet_wrap(~Samples, scales = 'free')+
  labs(title = 'Traveling Salesman Problem: Distance Traveled by Number of Stops')+
  theme_bw()

# Time charts
tsm_s = read_csv('travel_stats.csv')
om_s = read_csv('onemax_stats.csv')
fp_s = read_csv('fourpeaks_stats.csv')

stats <- tsm_s %>% rbind(om_s) %>% rbind(fp_s)

stats %>% 
  ggplot(aes(Length, Time, color = Algorithm))+
  geom_line(size = 1)+
  facet_wrap(~Problem)+
  labs(title = 'Computation Time by Length of Input Space')+
  scale_y_log10()+
  theme_bw()

# ANN graphs
ann_rhc = read_csv('ann_hillClimb.csv') %>% rename('Restart' = `...2`, 'Accuracy'=accuracy)
ann_ga = read_csv('ann_GeneticAlgorithm.csv')
ann_simann = read_csv('ann_SimAnneal.csv')

ann_simann %>% 
  ggplot(aes(reorder(DecayMethod, -Accuracy), Accuracy))+
  geom_col(fill = 'skyblue')+
  labs(title = 'Simulated Annealing ANN Accuracy by Decay Method',
       x = "Decay Method")+
  theme_bw()

ann_ga %>% 
  ggplot(aes(PopulationSize, Accuracy, col = as.factor(MutationRate)))+
  geom_line(size = 1)+
  labs(title = 'Genetic Algorithm ANN Accuracy By Population Size and Mutation Rate',
       color = 'Mutation Rate')+
  theme_bw()

rhc_ann_perf = ann_rhc %>% 
  ggplot(aes(Restart, Accuracy))+
  geom_line(size = 1)+
  labs(title = 'RHC ANN Accuracy - Number of Restarts')+
  theme_bw()

ann_rhc_time = ann_rhc %>% 
  ggplot(aes(Restart, time))+
  geom_line(size = 1)+
  labs(title = 'Random Hill Climb Time by Restarts',
       y = 'Time in Seconds',
       x = 'Restart Number')+
  theme_bw()
ggarrange(rhc_ann_perf, ann_rhc_time)
