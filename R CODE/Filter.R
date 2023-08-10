rm(list=ls())
gc()

library(tidyverse)
library(metR)
library(ggpubr)
library(lubridate)
library(ncdf4)

#ERSST
{
  dat <- read_csv('Datos/ERSST_ordered_cluster_analysis_Kaplan_grid_1942-2022.csv')
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/ERSST_cluster_4.nc')
  centroid_4 <- ncvar_get(nc)
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/ERSST_cluster_5.nc')
  centroid_5 <- ncvar_get(nc)
  
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/ERSST_preprocessed_SST_1905-2018.nc')
  sst <- ncvar_get(nc)

  cont <- 0
  lll <- list(6:8,1:3)
  for(cc in list(1:3,6:8)){
    cont <- cont + 1
    
      id_cc <- which(dat$cluster %in% cc)
      
      if(id_cc[1] == 1){
        id_cc <- id_cc[-1]
      }
      n_aux <- length(id_cc)
      if(id_cc[n_aux] == nrow(dat)){
        id_cc <- id_cc[-n_aux]
      }
      test <- (dat$cluster[id_cc+1] %in% c(4,5,lll[[cont]]) & 
        dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]]) ) 
    
      if(any(test)){
        id_test <- id_cc[which(test)]
        for(jj in id_test){
          dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
          dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
          if(dist_5 < dist_4){
            dat$cluster[jj] <- 5
          }else{
            dat$cluster[jj] <- 4
          }
        }
      }
    
    id_cc <- which(dat$cluster %in% cc)
    id_cc <- id_cc[which(dat$cluster[id_cc+1] %in% cc)]
    
    if(id_cc[1] == 1){
      id_cc <- id_cc[-1]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)){
      id_cc <- id_cc[-n_aux]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)-1){
      id_cc <- id_cc[-n_aux]
    }
    test <- dat$cluster[id_cc+2] %in% c(4,5,lll[[cont]]) & 
      dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]])
    if(any(test)){
      id_test <- id_cc[which(test)]
      for(jj in id_test){
        dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
        dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
        if(dist_5 < dist_4){
          dat$cluster[jj] <- 5
        }else{
          dat$cluster[jj] <- 4
        }
      }
    }
    
    id_cc <- which(dat$cluster %in% cc)
    
    if(id_cc[1] == 1){
      id_cc <- id_cc[-1]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)){
      id_cc <- id_cc[-n_aux]
    }
    test <- (dat$cluster[id_cc+1] %in% c(4,5,lll[[cont]]) & 
               dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]]) ) 
    
    if(any(test)){
      id_test <- id_cc[which(test)]
      for(jj in id_test){
        dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
        dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
        if(dist_5 < dist_4){
          dat$cluster[jj] <- 5
        }else{
          dat$cluster[jj] <- 4
        }
      }
    }
    
}
  
  write_csv(x = dat,
            file = 'Datos/ERSST_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv')
}

# COBE
{
  dat <- read_csv('Datos/COBE_cluster_analysis_Kaplan_grid_1942-2022.csv')
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/COBE_cluster_4.nc')
  centroid_4 <- ncvar_get(nc)
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/COBE_cluster_5.nc')
  centroid_5 <- ncvar_get(nc)
  
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/COBE_preprocessed_SST_1905-2018.nc')
  sst <- ncvar_get(nc)
  
  cont <- 0
  lll <- list(6:8,1:3)
  for(cc in list(1:3,6:8)){
    cont <- cont + 1
    
    id_cc <- which(dat$cluster %in% cc)
    
    if(id_cc[1] == 1){
      id_cc <- id_cc[-1]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)){
      id_cc <- id_cc[-n_aux]
    }
    test <- (dat$cluster[id_cc+1] %in% c(4,5,lll[[cont]]) & 
               dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]]) ) 
    
    if(any(test)){
      id_test <- id_cc[which(test)]
      for(jj in id_test){
        dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
        dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
        if(dist_5 < dist_4){
          dat$cluster[jj] <- 5
        }else{
          dat$cluster[jj] <- 4
        }
      }
    }
    
    id_cc <- which(dat$cluster %in% cc)
    id_cc <- id_cc[which(dat$cluster[id_cc+1] %in% cc)]
    
    if(id_cc[1] == 1){
      id_cc <- id_cc[-1]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)){
      id_cc <- id_cc[-n_aux]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)-1){
      id_cc <- id_cc[-n_aux]
    }
    test <- dat$cluster[id_cc+2] %in% c(4,5,lll[[cont]]) & 
      dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]])
    if(any(test)){
      id_test <- id_cc[which(test)]
      for(jj in id_test){
        dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
        dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
        if(dist_5 < dist_4){
          dat$cluster[jj] <- 5
        }else{
          dat$cluster[jj] <- 4
        }
      }
    }
    
    id_cc <- which(dat$cluster %in% cc)
    
    if(id_cc[1] == 1){
      id_cc <- id_cc[-1]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)){
      id_cc <- id_cc[-n_aux]
    }
    test <- (dat$cluster[id_cc+1] %in% c(4,5,lll[[cont]]) & 
               dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]]) ) 
    
    if(any(test)){
      id_test <- id_cc[which(test)]
      for(jj in id_test){
        dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
        dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
        if(dist_5 < dist_4){
          dat$cluster[jj] <- 5
        }else{
          dat$cluster[jj] <- 4
        }
      }
    }
    
  }
  
  write_csv(x = dat,
            file = 'Datos/COBE_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv')
}

#Hadley
{
  dat <- read_csv('Datos/HadISST_cluster_analysis_Kaplan_grid_1942-2022.csv')
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/HadISST_cluster_4.nc')
  centroid_4 <- ncvar_get(nc)
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/HadISST_cluster_5.nc')
  centroid_5 <- ncvar_get(nc)
  
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/HadISST_preprocessed_SST_1905-2018.nc')
  sst <- ncvar_get(nc)
  
  cont <- 0
  lll <- list(6:8,1:3)
  for(cc in list(1:3,6:8)){
    cont <- cont + 1
    
    id_cc <- which(dat$cluster %in% cc)
    
    if(id_cc[1] == 1){
      id_cc <- id_cc[-1]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)){
      id_cc <- id_cc[-n_aux]
    }
    test <- (dat$cluster[id_cc+1] %in% c(4,5,lll[[cont]]) & 
               dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]]) ) 
    
    if(any(test)){
      id_test <- id_cc[which(test)]
      for(jj in id_test){
        dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
        dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
        if(dist_5 < dist_4){
          dat$cluster[jj] <- 5
        }else{
          dat$cluster[jj] <- 4
        }
      }
    }
    
    id_cc <- which(dat$cluster %in% cc)
    id_cc <- id_cc[which(dat$cluster[id_cc+1] %in% cc)]
    
    if(id_cc[1] == 1){
      id_cc <- id_cc[-1]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)){
      id_cc <- id_cc[-n_aux]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)-1){
      id_cc <- id_cc[-n_aux]
    }
    test <- dat$cluster[id_cc+2] %in% c(4,5,lll[[cont]]) & 
      dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]])
    if(any(test)){
      id_test <- id_cc[which(test)]
      for(jj in id_test){
        dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
        dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
        if(dist_5 < dist_4){
          dat$cluster[jj] <- 5
        }else{
          dat$cluster[jj] <- 4
        }
      }
    }
    
    id_cc <- which(dat$cluster %in% cc)
    
    if(id_cc[1] == 1){
      id_cc <- id_cc[-1]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)){
      id_cc <- id_cc[-n_aux]
    }
    test <- (dat$cluster[id_cc+1] %in% c(4,5,lll[[cont]]) & 
               dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]]) ) 
    
    if(any(test)){
      id_test <- id_cc[which(test)]
      for(jj in id_test){
        dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
        dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
        if(dist_5 < dist_4){
          dat$cluster[jj] <- 5
        }else{
          dat$cluster[jj] <- 4
        }
      }
    }
    
  }
  
  write_csv(x = dat,
            file = 'Datos/HadISST_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv')
}

#Kaplan
{
  dat <- read_csv('Datos/Kaplan_cluster_analysis_Kaplan_grid_1942-2022.csv')
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/Kaplan_cluster_4.nc')
  centroid_4 <- ncvar_get(nc)
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/Kaplan_cluster_5.nc')
  centroid_5 <- ncvar_get(nc)
  
  nc <- nc_open(filename = 'Datos/Datos_filtrados_y_centroides/Kaplan_preprocessed_SST_1905-2018.nc')
  sst <- ncvar_get(nc)
  
  cont <- 0
  lll <- list(6:8,1:3)
  for(cc in list(1:3,6:8)){
    cont <- cont + 1
    
    id_cc <- which(dat$cluster %in% cc)
    
    if(id_cc[1] == 1){
      id_cc <- id_cc[-1]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)){
      id_cc <- id_cc[-n_aux]
    }
    test <- (dat$cluster[id_cc+1] %in% c(4,5,lll[[cont]]) & 
               dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]]) ) 
    
    if(any(test)){
      id_test <- id_cc[which(test)]
      for(jj in id_test){
        dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
        dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
        if(dist_5 < dist_4){
          dat$cluster[jj] <- 5
        }else{
          dat$cluster[jj] <- 4
        }
      }
    }
    
    id_cc <- which(dat$cluster %in% cc)
    id_cc <- id_cc[which(dat$cluster[id_cc+1] %in% cc)]
    
    if(id_cc[1] == 1){
      id_cc <- id_cc[-1]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)){
      id_cc <- id_cc[-n_aux]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)-1){
      id_cc <- id_cc[-n_aux]
    }
    test <- dat$cluster[id_cc+2] %in% c(4,5,lll[[cont]]) & 
      dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]])
    if(any(test)){
      id_test <- id_cc[which(test)]
      for(jj in id_test){
        dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
        dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
        if(dist_5 < dist_4){
          dat$cluster[jj] <- 5
        }else{
          dat$cluster[jj] <- 4
        }
      }
    }
    
    id_cc <- which(dat$cluster %in% cc)
    
    if(id_cc[1] == 1){
      id_cc <- id_cc[-1]
    }
    n_aux <- length(id_cc)
    if(id_cc[n_aux] == nrow(dat)){
      id_cc <- id_cc[-n_aux]
    }
    test <- (dat$cluster[id_cc+1] %in% c(4,5,lll[[cont]]) & 
               dat$cluster[id_cc-1] %in% c(4,5,lll[[cont]]) ) 
    
    if(any(test)){
      id_test <- id_cc[which(test)]
      for(jj in id_test){
        dist_4 <- sum( (sst[,,jj] - centroid_4)**2)
        dist_5 <- sum( (sst[,,jj] - centroid_5)**2)
        if(dist_5 < dist_4){
          dat$cluster[jj] <- 5
        }else{
          dat$cluster[jj] <- 4
        }
      }
    }
    
  }
  
  write_csv(x = dat,
            file = 'Datos/Kaplan_ordered_cluster_analysis_Kaplan_grid_1942-2022_FILTER.csv')
}
