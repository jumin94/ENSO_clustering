rm(list=ls())
gc()

library(tidyverse)
library(metR)
library(ggpubr)
library(lubridate)
library(ggforce)
library(ggh4x)

# Fig 3, 4 and 6
{
  {
    datos <- read_csv(file = 'Datos/ERSST_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
      select(time, cluster) %>%
      rename(ERSST = cluster) %>%
      left_join(
        read_csv(file = 'Datos/Kaplan_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
          select(time, cluster) %>%
          rename(KAPLAN = cluster)
      ) %>%
      left_join(
        read_csv(file = 'Datos/COBE_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
          select(time, cluster) %>%
          rename(COBE = cluster)
      ) %>%
      mutate(YY = year(time), MM = month(time)) %>%
      select(-time) %>%
      left_join(
        read_csv(file = 'Datos/HadISST_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
          select(time, cluster) %>%
          rename(HADLEY = cluster)%>%
          mutate(YY = year(time), MM = month(time)) %>%
          select(-time)
      )
    
    
    mean_length <- data.frame(Cluster= factor(NA, levels = c('C1','C2','C3',
                                                             'C4','C5','C6',
                                                             'C7','C8')),
                              DATA = 
                                c('COBE',
                                  'HADLEY',
                                  'ERSST',
                                  'KAPLAN'),
                              Length = NA
    )
    
    for(cc in 1:8){
      
      dat <- datos
      dat$cc <- datos$COBE
      
      {
        id_ini <- which(dat$cc == cc) 
        if(id_ini[1]==1){
          id_ini <- c(1,id_ini[which(dat$cc[id_ini-1] != cc)+1])
        }else{
          id_ini <- id_ini[which(dat$cc[id_ini-1] != cc)]
        }
        
        aux_vec <- vector(mode = 'double', length = length(id_ini))
        cont_for <- 0
        for(ii in id_ini){
          cont_for <- cont_for + 1
          cont <- 1
          aux_cond <- dat$cc[ii+cont] == cc
          while (aux_cond) {
            cont <- cont + 1
            if(ii+cont == nrow(datos)+1){break}
            aux_cond <- dat$cc[ii+cont] == cc
          }
          aux_vec[cont_for] <- cont
        }
      }
      
      mean_length <- rbind(mean_length,
                           data.frame(Cluster= paste0('C',cc),
                                      DATA = 'COBE',
                                      Length = aux_vec))
      
      dat <- datos
      dat$cc <- datos$ERSST
      
      {
        id_ini <- which(dat$cc == cc) 
        if(id_ini[1]==1){
          id_ini <- c(1,id_ini[which(dat$cc[id_ini-1] != cc)+1])
        }else{
          id_ini <- id_ini[which(dat$cc[id_ini-1] != cc)]
        }
        
        aux_vec <- vector(mode = 'double', length = length(id_ini))
        cont_for <- 0
        for(ii in id_ini){
          cont_for <- cont_for + 1
          cont <- 1
          aux_cond <- dat$cc[ii+cont] == cc
          while (aux_cond) {
            cont <- cont + 1
            if(ii+cont == nrow(datos)+1){break}
            aux_cond <- dat$cc[ii+cont] == cc
          }
          aux_vec[cont_for] <- cont
        }
      }
      mean_length <- rbind(mean_length,
                           data.frame(Cluster= paste0('C',cc),
                                      DATA = 'ERSST',
                                      Length = aux_vec))
      
      
      dat <- datos
      dat$cc <- datos$HADLEY
      
      {
        id_ini <- which(dat$cc == cc) 
        if(id_ini[1]==1){
          id_ini <- c(1,id_ini[which(dat$cc[id_ini-1] != cc)+1])
        }else{
          id_ini <- id_ini[which(dat$cc[id_ini-1] != cc)]
        }
        
        aux_vec <- vector(mode = 'double', length = length(id_ini))
        cont_for <- 0
        for(ii in id_ini){
          cont_for <- cont_for + 1
          cont <- 1
          aux_cond <- dat$cc[ii+cont] == cc
          while (aux_cond) {
            cont <- cont + 1
            if(ii+cont == nrow(datos)+1){break}
            aux_cond <- dat$cc[ii+cont] == cc
          }
          aux_vec[cont_for] <- cont
        }
      }
      mean_length <- rbind(mean_length,
                           data.frame(Cluster= paste0('C',cc),
                                      DATA = 'HADLEY',
                                      Length = aux_vec))
      
      
      dat <- datos
      dat$cc <- datos$KAPLAN
      
      {
        id_ini <- which(dat$cc == cc) 
        if(id_ini[1]==1){
          id_ini <- c(1,id_ini[which(dat$cc[id_ini-1] != cc)+1])
        }else{
          id_ini <- id_ini[which(dat$cc[id_ini-1] != cc)]
        }
        
        aux_vec <- vector(mode = 'double', length = length(id_ini))
        cont_for <- 0
        for(ii in id_ini){
          cont_for <- cont_for + 1
          cont <- 1
          aux_cond <- dat$cc[ii+cont] == cc
          while (aux_cond) {
            cont <- cont + 1
            if(ii+cont == nrow(datos)+1){break}
            aux_cond <- dat$cc[ii+cont] == cc
          }
          aux_vec[cont_for] <- cont
        }
      }
      mean_length <- rbind(mean_length,
                           data.frame(Cluster= paste0('C',cc),
                                      DATA = 'KAPLAN',
                                      Length = aux_vec))
    }
  }
  
  mean_length <- mean_length %>%
    mutate(DATA = case_when(
      DATA == 'KAPLAN' ~ 'Kaplan',
      DATA == 'HADLEY' ~ 'HadISST',
      T ~ DATA
    ))
  ggsave(filename = 'Fig_6.tiff', 
         width = 3300, height = 2400,
         units = 'px',
         bg = 'white',
         plot = 
           ggplot(data = mean_length %>% drop_na(),
                  mapping = aes(x = Cluster, y = Length,
                                fill = Cluster)) +
           theme_bw() +
           geom_violin(draw_quantiles = c(0.25,0.5,0.75)) +
           scale_fill_manual(values = c(colorRampPalette(c(scales::muted('blue'),'white'))(4)[1:3],
                                        'green','yellow',
                                        colorRampPalette(c('white',scales::muted('red')))(4)[2:4]
           )) +
           facet_wrap(~DATA)
         
  )
  
  ggsave(filename = 'Fig_3.tiff', 
         width = 3300, height = 2400,
         units = 'px',
         bg = 'white',
         plot = 
           datos %>%
           mutate(MM = factor(MM, levels = 1:12)) %>%
           gather(key = DATA, value = Cluster, - YY, - MM) %>%
           filter(Cluster %in% c(1:3,6:8)) %>%
           mutate(DATA = case_when(
             DATA == 'KAPLAN' ~ 'Kaplan',
             DATA == 'HADLEY' ~ 'HadISST',
             T ~ DATA
           )) %>%
           group_by(DATA, Cluster, MM) %>%
           count() %>%
           mutate(n = ifelse(Cluster %in% 1:3, -n,n)) %>%
           mutate(Cluster = factor(paste0('C',Cluster), 
                                   levels = c('C1','C2','C3',
                                              'C6',
                                              'C7','C8') )) %>%
           ggplot(mapping = aes(x = MM, fill = Cluster, y = n)) +
           theme_bw() +
           geom_bar(data = .%>% filter(!Cluster %in% c('C1','C2','C3')),
                    stat="identity",position = position_stack(reverse = FALSE)) +
           geom_bar(data = .%>% filter(Cluster %in% c('C1','C2','C3')),
                    stat="identity",position = position_stack(reverse = TRUE)) +
           facet_wrap(~DATA) +
           scale_fill_manual(values = c(colorRampPalette(c(scales::muted('blue'),'white'))(4)[1:3],
                                        colorRampPalette(c('white',scales::muted('red')))(4)[2:4])) +
           xlab('Month') +
           scale_y_continuous(name = 'Count', 
                              breaks = c(-50,-25,0,25,50),
                              labels = c(50,25,0,25,50))
         
  )
  
  ggsave(filename = 'Fig_4.tiff', 
         width = 3300, height = 2400,
         units = 'px',
         bg = 'white',
         plot = 
           datos %>%
           mutate(MM = factor(MM, levels = 1:12)) %>%
           gather(key = DATA, value = Cluster, - YY, - MM) %>%
           mutate(DATA = case_when(
             DATA == 'KAPLAN' ~ 'Kaplan',
             DATA == 'HADLEY' ~ 'HadISST',
             T ~ DATA
           )) %>%
           filter(Cluster %in% 4:5) %>%
           group_by(DATA, Cluster, MM) %>%
           count() %>%
           mutate(n = ifelse(Cluster == 4, -n,n)) %>%
           mutate(Cluster = factor(paste0('C',Cluster), 
                                   levels = c('C4','C5') )) %>%
           ggplot(mapping = aes(x = MM, fill = Cluster, y = n)) +
           theme_bw() +
           geom_bar(stat="identity") +
           facet_wrap(~DATA) +
           scale_fill_manual(values = c('green','yellow')) +
           xlab('Month') +
           scale_y_continuous(name = 'Count', 
                              breaks = c(-50,-25,0,25,50),
                              labels = c(50,25,0,25,50))
         
  )
}

# Fig 5
{
  {
  {
    dat <- read_csv(file = 'Datos/HadISST_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
      select(time, cluster)
    transisiones <- data.frame(Ini= c('C1','C2','C3',
                                      'C4','C5','C6',
                                      'C7','C8'),
                               C1 = NA, C2 = NA, C3 = NA,
                               C4 = NA, C5 = NA, C6 = NA,
                               C7 = NA, C8 = NA)
    Freq <-  data.frame(Ini= c('C1','C2','C3',
                               'C4','C5','C6',
                               'C7','C8'), Freq = NA)
    
    for(ii in 1:8){
      Freq[ii,2] <- length(which(dat$cluster == ii))
      for(jj in 1:8){
        
        id <- which(dat$cluster == ii)
        if(any(dat$cluster[id+1] == jj, na.rm = T)){
          transisiones[ii,jj+1] <- sum(dat$cluster[id+1] == jj, na.rm = T)
        }else{
          transisiones[ii,jj+1] <- 0
        }
      }
    }
    Freq$DATA <- 'HADLEY'
    
    Freq1 <- Freq
    
    transisiones_1 <- transisiones %>%
      gather(key = To, value = Count, - Ini) %>%
      group_by(Ini) %>%
      mutate(Freq = Count/sum(Count)) %>%
      ungroup() %>%
      mutate(DATA = 'HADLEY')
  }
  {
    dat <- read_csv(file = 'Datos/Kaplan_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
      select(time, cluster)
    transisiones <- data.frame(Ini= c('C1','C2','C3',
                                      'C4','C5','C6',
                                      'C7','C8'),
                               C1 = NA, C2 = NA, C3 = NA,
                               C4 = NA, C5 = NA, C6 = NA,
                               C7 = NA, C8 = NA)
    Freq <-  data.frame(Ini= c('C1','C2','C3',
                               'C4','C5','C6',
                               'C7','C8'), Freq = NA)
    
    for(ii in 1:8){
      Freq[ii,2] <- length(which(dat$cluster == ii))
      for(jj in 1:8){
        
        id <- which(dat$cluster == ii)
        if(any(dat$cluster[id+1] == jj, na.rm = T)){
          transisiones[ii,jj+1] <- sum(dat$cluster[id+1] == jj, na.rm = T)
        }else{
          transisiones[ii,jj+1] <- 0
        }
      }
    }
    Freq$DATA <- 'KAPLAN'
    
    Freq2 <- Freq
    
    transisiones_2 <- transisiones %>%
      gather(key = To, value = Count, - Ini) %>%
      group_by(Ini) %>%
      mutate(Freq = Count/sum(Count)) %>%
      ungroup() %>%
      mutate(DATA = 'KAPLAN')
  }
  {
    dat <- read_csv(file = 'Datos/COBE_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
      select(time, cluster)
    transisiones <- data.frame(Ini= c('C1','C2','C3',
                                      'C4','C5','C6',
                                      'C7','C8'),
                               C1 = NA, C2 = NA, C3 = NA,
                               C4 = NA, C5 = NA, C6 = NA,
                               C7 = NA, C8 = NA)
    Freq <-  data.frame(Ini= c('C1','C2','C3',
                               'C4','C5','C6',
                               'C7','C8'), Freq = NA)
    
    for(ii in 1:8){
      Freq[ii,2] <- length(which(dat$cluster == ii))
      for(jj in 1:8){
        
        id <- which(dat$cluster == ii)
        if(any(dat$cluster[id+1] == jj, na.rm = T)){
          transisiones[ii,jj+1] <- sum(dat$cluster[id+1] == jj, na.rm = T)
        }else{
          transisiones[ii,jj+1] <- 0
        }
      }
    }
    Freq$DATA <- 'COBE'
    
    Freq3 <- Freq
    
    transisiones_3 <- transisiones %>%
      gather(key = To, value = Count, - Ini) %>%
      group_by(Ini) %>%
      mutate(Freq = Count/sum(Count)) %>%
      ungroup() %>%
      mutate(DATA = 'COBE')
  }
  {
    dat <- read_csv(file = 'Datos/ERSST_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
      select(time, cluster)
    transisiones <- data.frame(Ini= c('C1','C2','C3',
                                      'C4','C5','C6',
                                      'C7','C8'),
                               C1 = NA, C2 = NA, C3 = NA,
                               C4 = NA, C5 = NA, C6 = NA,
                               C7 = NA, C8 = NA)
    Freq <-  data.frame(Ini= c('C1','C2','C3',
                               'C4','C5','C6',
                               'C7','C8'), Freq = NA)
    
    for(ii in 1:8){
      Freq[ii,2] <- length(which(dat$cluster == ii))
      for(jj in 1:8){
        
        id <- which(dat$cluster == ii)
        if(any(dat$cluster[id+1] == jj, na.rm = T)){
          transisiones[ii,jj+1] <- sum(dat$cluster[id+1] == jj, na.rm = T)
        }else{
          transisiones[ii,jj+1] <- 0
        }
      }
    }
    Freq$DATA <- 'ERSST'
    
    Freq4 <- Freq
    
    transisiones_4 <- transisiones %>%
      gather(key = To, value = Count, - Ini) %>%
      group_by(Ini) %>%
      mutate(Freq = Count/sum(Count)) %>%
      ungroup() %>%
      mutate(DATA = 'ERSST')
  }
  
  
  freq <- rbind(Freq1,Freq2,Freq3,Freq4)
  transisiones <- rbind(transisiones_1,transisiones_2,
                        transisiones_3,transisiones_4)
  
  
  write.csv(x = transisiones,
            file = 'TRANSITIONS.csv')
  write.csv(x = freq,
            file = 'FREQ.csv')
  
  
  freq <- read_csv(file = 'FREQ.csv')
  transisiones <- read_csv(file = 'TRANSITIONS.csv')
  }
  transisiones <- transisiones %>%
    mutate(DATA = case_when(
      DATA == 'KAPLAN' ~ 'Kaplan',
      DATA == 'HADLEY' ~ 'HadISST',
      T ~ DATA
    ))
  
  ggsave(filename = 'Fig_5.tiff', 
         width = 3600, height = 3600,
         units = 'px',
         plot = transisiones %>%
           mutate(Freq = Freq * 100) %>%
           ggplot(mapping = aes(x = To, y = Ini)) +
           theme_bw() +
           geom_tile(data = .%>%filter(To != Ini),
                     mapping = aes(fill = Freq),
                     col = 'black') +
           geom_tile(data = .%>%filter(To == Ini), fill = 'gray') +
           geom_text(data = .%>%filter(Freq != 0),
                     mapping = aes(label = round(Freq))) +
           scale_fill_divergent('Freq %', limits = c(0,35)) +
           xlab('To') + ylab('From') +
           facet_wrap(~DATA)
  )
}

# Fig 7
{
  {
  # Transisiones 2
  transisiones <- data.frame(From= c('C1'),
                             To= c('C1'),
                             Month = 1,
                             Count = NA,
                             DATA = 'ERSST')
  {
    dat <- read_csv(file = 'Datos/HadISST_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
      select(time, cluster)
    
    transisiones_aux <- data.frame(From= rep(c('C1'),8*8*12),
                                   To= c('C1'),
                                   Month = 1,
                                   Count = NA,
                                   DATA = 'ERSST')
    
    count <- 0
    meses <- list(
      c(12,1:2),
      1:3,
      2:4,
      3:5,
      4:6,
      5:7,
      6:8,
      7:9,
      8:10,
      9:11,
      10:12,
      c(11:12,1)
    )
    for(mm in 1:12){
      dat_id <- which(month(dat$time) %in% meses[[mm]])
      if(dat_id[length(dat_id)] == nrow(dat)){
        dat_id <- dat_id[-length(dat_id)] 
      }
      dat_aux <- dat[sort(c(dat_id,dat_id+1)),]
      
      for(ii in 1:8){
        Freq_aux<- length(which(dat$cluster[dat_id] == ii))
        id <- which(dat_aux$cluster == ii & 
                      month(dat_aux$time) %in%meses[[mm]])
        
        for(jj in 1:8){
          count <- count + 1
          if(any(dat_aux$cluster[id+1] == jj, na.rm = T)){
            transisiones_aux$Count[count] <- sum(dat_aux$cluster[id+1] == jj, na.rm = T)/Freq_aux
          }else{
            transisiones_aux$Count[count] <- 0
          }
          transisiones_aux$From[count] <- paste0('C',ii)
          transisiones_aux$To[count] <- paste0('C',jj)
          transisiones_aux$Month[count] <- mm
          
        }
      }
    }
    transisiones_aux$DATA <- 'HADLEY'
    transisiones <- rbind(transisiones, 
                          transisiones_aux)
  }
  {
    dat <- read_csv(file = 'Datos/Kaplan_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
      select(time, cluster)
    transisiones_aux <- data.frame(From= rep(c('C1'),8*8*12),
                                   To= c('C1'),
                                   Month = 1,
                                   Count = NA,
                                   DATA = 'ERSST')
    
    count <- 0
    meses <- list(
      c(12,1:2),
      1:3,
      2:4,
      3:5,
      4:6,
      5:7,
      6:8,
      7:9,
      8:10,
      9:11,
      10:12,
      c(11:12,1)
    )
    for(mm in 1:12){
      dat_id <- which(month(dat$time) %in% meses[[mm]])
      if(dat_id[length(dat_id)] == nrow(dat)){
        dat_id <- dat_id[-length(dat_id)] 
      }
      dat_aux <- dat[sort(c(dat_id,dat_id+1)),]
      
      for(ii in 1:8){
        Freq_aux<- length(which(dat$cluster[dat_id] == ii))
        id <- which(dat_aux$cluster == ii & 
                      month(dat_aux$time) %in%meses[[mm]])
        
        for(jj in 1:8){
          count <- count + 1
          if(any(dat_aux$cluster[id+1] == jj, na.rm = T)){
            transisiones_aux$Count[count] <- sum(dat_aux$cluster[id+1] == jj, na.rm = T)/Freq_aux
          }else{
            transisiones_aux$Count[count] <- 0
          }
          transisiones_aux$From[count] <- paste0('C',ii)
          transisiones_aux$To[count] <- paste0('C',jj)
          transisiones_aux$Month[count] <- mm
          
        }
      }
    }
    transisiones_aux$DATA <- 'KAPLAN'
    transisiones <- rbind(transisiones, 
                          transisiones_aux)
  }
  {
    dat <- read_csv(file = 'Datos/COBE_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
      select(time, cluster)
    transisiones_aux <- data.frame(From= rep(c('C1'),8*8*12),
                                   To= c('C1'),
                                   Month = 1,
                                   Count = NA,
                                   DATA = 'ERSST')
    
    count <- 0
    meses <- list(
      c(12,1:2),
      1:3,
      2:4,
      3:5,
      4:6,
      5:7,
      6:8,
      7:9,
      8:10,
      9:11,
      10:12,
      c(11:12,1)
    )
    for(mm in 1:12){
      dat_id <- which(month(dat$time) %in% meses[[mm]])
      if(dat_id[length(dat_id)] == nrow(dat)){
        dat_id <- dat_id[-length(dat_id)] 
      }
      dat_aux <- dat[sort(c(dat_id,dat_id+1)),]
      
      for(ii in 1:8){
        Freq_aux<- length(which(dat$cluster[dat_id] == ii))
        id <- which(dat_aux$cluster == ii & 
                      month(dat_aux$time) %in%meses[[mm]])
        
        for(jj in 1:8){
          count <- count + 1
          if(any(dat_aux$cluster[id+1] == jj, na.rm = T)){
            transisiones_aux$Count[count] <- sum(dat_aux$cluster[id+1] == jj, na.rm = T)/Freq_aux
          }else{
            transisiones_aux$Count[count] <- 0
          }
          transisiones_aux$From[count] <- paste0('C',ii)
          transisiones_aux$To[count] <- paste0('C',jj)
          transisiones_aux$Month[count] <- mm
          
        }
      }
    }
    transisiones_aux$DATA <- 'COBE'
    transisiones <- rbind(transisiones, 
                          transisiones_aux)
  }
  {
    dat <- read_csv(file = 'Datos/ERSST_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv') %>%
      select(time, cluster)
    transisiones_aux <- data.frame(From= rep(c('C1'),8*8*12),
                                   To= c('C1'),
                                   Month = 1,
                                   Count = NA,
                                   DATA = 'ERSST')
    
    count <- 0
    meses <- list(
      c(12,1:2),
      1:3,
      2:4,
      3:5,
      4:6,
      5:7,
      6:8,
      7:9,
      8:10,
      9:11,
      10:12,
      c(11:12,1)
    )
    for(mm in 1:12){
      dat_id <- which(month(dat$time) %in% meses[[mm]])
      if(dat_id[length(dat_id)] == nrow(dat)){
        dat_id <- dat_id[-length(dat_id)] 
      }
      dat_aux <- dat[sort(c(dat_id,dat_id+1)),]
      
      for(ii in 1:8){
        Freq_aux<- length(which(dat$cluster[dat_id] == ii))
        id <- which(dat_aux$cluster == ii & 
                      month(dat_aux$time) %in%meses[[mm]])
        
        for(jj in 1:8){
          count <- count + 1
          if(any(dat_aux$cluster[id+1] == jj, na.rm = T)){
            transisiones_aux$Count[count] <- sum(dat_aux$cluster[id+1] == jj, na.rm = T)/Freq_aux
          }else{
            transisiones_aux$Count[count] <- 0
          }
          transisiones_aux$From[count] <- paste0('C',ii)
          transisiones_aux$To[count] <- paste0('C',jj)
          transisiones_aux$Month[count] <- mm
          
        }
      }
    }
    transisiones_aux$DATA <- 'ERSST'
    transisiones <- rbind(transisiones, 
                          transisiones_aux)
  }
  }
  transisiones <- transisiones %>%
    mutate(DATA = case_when(
      DATA == 'KAPLAN' ~ 'Kaplan',
      DATA == 'HADLEY' ~ 'HadISST',
      T ~ DATA
    ))
  ggsave(filename = 'Fig_7.tiff', 
         width = 6000, height = 4800,
         units = 'px',
         bg = 'white',
         plot = 
           transisiones %>%
           mutate(From = factor(From,
                                levels = c('C1','C2','C3',
                                           'C4','C5','C6',
                                           'C7','C8'))) %>%
           mutate(To = factor(To,
                              levels = c('C1','C2','C3',
                                         'C4','C5','C6',
                                         'C7','C8'))) %>%
           mutate(Month = factor(Month, levels = 1:12)) %>%
           mutate(Count = 100* Count) %>%
           group_by(From,To,Month) %>%
           summarise(Mean = mean(Count),
                     Uncertainty = (max(Count) - min(Count))/2) %>%
           mutate(Mean = factor(
             case_when(
               Mean == 0 ~ 'NA',
               Mean < 5 ~ '<5%',
               Mean < 15 ~ '5%-15%',
               Mean < 25 ~ '15%-25%',
               Mean < 35 ~ '25%-35%',
               Mean < 45 ~ '35%-45%',
               Mean < 55 ~ '45%-55%',
               T ~ '55%<'
             ), levels = c('<5%',
                           '5%-15%','15%-25%','25%-35%',
                           '35%-45%','45%-55%','55%<')
           )) %>%
           ggplot(mapping = aes(x = Month, y = To,
                                label = round(Uncertainty),
                                fill = Mean)) +
           theme_bw() +
           geom_tile(data = .%>% filter(Mean != 'NA'),
                     col = 'black') +
           geom_text(data = .%>% filter(Mean != 'NA')) +
           scale_fill_manual(name = 'Mean(Freq)', 
                             values =  c(
                               colorRampPalette(c("white",
                                                  "#00FFFFFF",
                                                  "#80FF00FF",
                                                  'yellow',
                                                  'orange',
                                                  "#FF0000FF",
                                                  "#8000FFFF"))(8)[-1]
                               # colorRampPalette(c('white','yellow'))(5)[-1],
                               # colorRampPalette(c('yellow',scales::muted('red')))(4)[2:4]
                             )
           ) +
           facet_wrap(~From)
  )
}

# Fig 8
{
  {
    datos <- read.table(file = 'ENSO-ClassicalMethodology-X-Clusters.csv', 
                        stringsAsFactors = F,
                        sep = ';', header = T)[-1,]
    datos$Date <- dmy(datos$Date)
    datos <- datos %>% filter(year(Date) %in% 1942:2018)
    
    datos <- datos[,c(1,3,5,7,9)]
    colnames(datos)[2:5] <- c('COBE','ERSST','HADLEY','KAPLAN')
    
    datos <- datos %>% 
      gather(key = DB, value = Cluster, -Date) %>%
      mutate(MM = month(Date), YY = year(Date)) %>%
      filter(MM %in% c(1:2,9:12)) %>%
      mutate(Event = ifelse(MM < 3,
                            paste0(YY-1,'-',YY),
                            paste0(YY,'-',YY+1)))
    
    EN_PERIOD <- datos %>%
      filter(MM == 12) %>%
      filter(Cluster %in% c(6:8)) %>%
      mutate(Cluster12 = Cluster) %>%
      select(Cluster12, Event, DB) %>%
      mutate(ENSO = 'EN')
    LN_PERIOD <- datos %>%
      filter(MM == 12) %>%
      filter(Cluster %in% c(1:3)) %>%
      mutate(Cluster12 = Cluster) %>%
      select(Cluster12, Event, DB) %>%
      mutate(ENSO = 'LN')
    
    graf_guard <- rbind(LN_PERIOD %>%
                          left_join(datos),
                        EN_PERIOD %>%
                          left_join(datos)) %>%
      filter(DB == 'ERSST') %>%
      mutate(Month = factor(MM, levels = c(9:12,1:2))) %>%
      mutate(Cluster = factor(Cluster, levels = 1:8)) %>%
      ggplot(mapping = aes(x = Month, y = Event, fill = Cluster)) +
      theme_bw() +
      geom_tile() +
      facet_nested(ENSO + Cluster12 ~.,
                   scales = 'free')+
      force_panelsizes(rows =  c(0.7,
                                 1.5,
                                 0.4,
                                 1.5,
                                 1.2,
                                 0.6)) +
      scale_fill_manual(na.value = 'black',
                        values = c(colorRampPalette(c(scales::muted('blue'),'white'))(4)[1:3],
                                   'green','yellow',
                                   colorRampPalette(c('white',scales::muted('red')))(4)[2:4]))
  }

  {
    datos <- read.table(file = 'ENSO-ClassicalMethodology-X-Clusters.csv', 
                        stringsAsFactors = F,
                        sep = ';', header = T)[-1,]
    datos$Date <- dmy(datos$Date)
    datos <- datos %>% filter(year(Date) %in% 1942:2018) %>%
      filter(Date < ymd('2018-03-03'))
    
    datos <- datos[,c(1,2,4,6,8)]
    colnames(datos)[2:5] <- c('COBE','ERSST','HADLEY','KAPLAN')
    
    datos <- datos %>% 
      gather(key = DB, value = Cluster, -Date) %>%
      mutate(MM = month(Date), YY = year(Date)) %>%
      filter(MM %in% c(1:2,9:12)) %>%
      mutate(Event = ifelse(MM < 3,
                            paste0(YY-1,'-',YY),
                            paste0(YY,'-',YY+1))) %>%
      mutate(ENSO = case_when(Cluster %in% c('EN123', 
                                             'EN3') ~ 'EN3',
                              Cluster %in% c('EN124',
                                             'EN4','EN4LN12') ~ 'EN4',
                              Cluster %in% c('EN34','EN1234') ~ 'EN34',
                              Cluster %in% c('LN3','LN123') ~ 'LN3',
                              Cluster %in% c('LN4','LN124',
                                             'EN12LN4') ~ 'LN4',
                              Cluster %in% c('LN34','LN1234',
                                             'EN12LN34') ~ 'LN34',
                              Cluster %in% c('EN12',
                                             'NEU',
                                             'LN12') ~ 'Neu')) %>%
      mutate(Coast = case_when(Cluster %in% c('EN123','EN1234',
                                              'EN124','EN12','EN12LN4',
                                              'EN12LN34') ~ 'EN',
                               Cluster %in% c('EN4LN12','LN124',
                                              'LN123','LN12',
                                              'LN1234') ~ 'LN',
                               T ~ 'Neu'))
    
    EN_PERIOD <- datos %>%
      filter(MM == 12) %>%
      filter(ENSO %in% c('EN3','EN4','EN34')) %>%
      mutate(Cluster12 = ENSO) %>%
      select(Cluster12, Event, DB) %>%
      mutate(facet = 'EN')
    LN_PERIOD <- datos %>%
      filter(MM == 12) %>%
      filter(ENSO %in% c('LN3','LN4','LN34')) %>%
      mutate(Cluster12 = ENSO) %>%
      select(Cluster12, Event, DB) %>%
      mutate(facet = 'LN')
    
    yy1 <- c(1918, 1925, 1930,
             1940, 1941, 1957,
             1965, 1969, 1972,
             1982, 1991, 1997,
             2015, 1916, 1949, 1954,
             1955, 1970, 1973,
             1975, 1999, 2007,
             2010, 2020)
    yy2 <- c(1968, 1986, 1987,
             2002, 2006, 2009,
             2018, 1917, 1924, 1933,
             1984, 1988, 1998,
             2011)
    yy3 <- c(1919, 1939, 1951,
             1963, 1976, 1942, 1967, 1971,
             1985, 1995, 1996,
             2005, 2017)
    yy4 <- c(1923, 1977, 1990,
             1994, 2004, 2014,
             2019, 1915, 1938, 1950,
             1964, 1983, 2000,
             2008) 
    graf_box <- rbind(
      EN_PERIOD %>%
        left_join(datos),
      LN_PERIOD %>%
        left_join(datos)) %>%
      filter(DB == 'ERSST') %>%
      mutate(Month = factor(MM, levels = c(9:12,1:2))) %>%
      mutate(ENSO = factor(ENSO, levels = c('LN34','LN4','LN3',
                                            'Neu',
                                            'EN3','EN4','EN34'))) %>%
      mutate(cluster = case_when(
        Event %in% paste0(yy1,'-',yy1+1) ~ 'Spread',
        Event %in% paste0(yy2,'-',yy2+1) ~ 'Central+East',
        Event %in% paste0(yy3,'-',yy3+1) ~ 'East',
        Event %in% paste0(yy4,'-',yy4+1) ~ 'Central'
      )) %>%
      ggplot(mapping = aes(x = Month, y = Event)) +
      theme_bw() +
      geom_tile(mapping = aes(fill = ENSO)) +
      geom_point(data = .%>% filter(Coast == 'LN'),
                 shape = 6) +
      geom_point(data = .%>% filter(Coast == 'EN'),
                 shape = 2) +
      facet_nested(facet + cluster  ~.,
                   scales = 'free')+
      force_panelsizes(rows =  c(0.2,
                                 2.1,
                                 0.3,
                                 1.0,
                                 1.9,
                                 0.2)) +
      guides(shape = F) +
      scale_fill_manual(na.value = 'black',
                        values = c(colorRampPalette(c(scales::muted('blue'),'white'))(4)[1:3],
                                   'green',
                                   colorRampPalette(c('white',scales::muted('red')))(4)[2:4]))
  }
  
  ggsave(filename = 'Fig_8.tiff', 
         width = 6000, height = 4800,
         units = 'px',
         bg = 'white',
         plot = 
           ggarrange(graf_guard, graf_box
           )
  )
  
}