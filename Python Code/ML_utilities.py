#ML toolbox

# Common imports
import numpy as np
import pandas as pd
import os
import xarray as xr

# To plot pretty figures
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import sys

#Clustering imports
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def preprocessing(PathAndFilename,subset=False):
    '''
        If selecting a specific period, use subset=[yr0,yrf] or subset = [yr0]
    '''

    ds = xr.open_dataset(PathAndFilename)
    if('Had' in PathAndFilename):
        print('econtro esto')
        #Fixing coord system from -180~180 to 0~360
        if 'tos' in ds:
          tmp=xr.concat([ds.tos.sel(longitude=slice(0,180)),ds.tos.sel(longitude=slice(-180,0))],dim='longitude')
        else:
          tmp=xr.concat([ds.sst.sel(longitude=slice(0,180)),ds.sst.sel(longitude=slice(-180,0))],dim='longitude')
        tmp['longitude']=np.where(tmp.longitude.values < 0,tmp.longitude+360,tmp.longitude.values)
        sst = tmp.rename({'longitude':'lon','latitude':'lat'})
        if sst.lat[0] >= sst.lat[1]:
          sst = sst.sel(lon=slice(140,280), lat=slice(15,-15)) # subset the data
        else:
          sst = sst.sel(lon=slice(140,280), lat=slice(-15,15)) # subset the data
    else:
      if ds.lat[0] >= ds.lat[1]:
        sst = ds.sst.sel(lon=slice(140,280), lat=slice(15,-15)) # subset the data
      else:
        sst = ds.sst.sel(lon=slice(140,280), lat=slice(-15,15)) # subset the data

    #subseting in time:
    if subset:
        if(len(subset) > 1):
            sst=sst.sel(time=slice(str(subset[0]),str(subset[1])))
        else:
            sst=sst.where(sst['time.year'] > subset[0],drop=True) 
    
    # preprocessing
    ssta = sst.groupby('time.month') - sst.groupby('time.month').mean(dim='time', skipna=True) # remove monthly variability
    sstd = ssta - ssta.rolling(time=120, center=True).mean() # removing trend
    sstf = sstd.dropna('time', how='all') # dropping NaN time steps resulting from rolling mean.
    
    # index 3.4
    if sst.lat[0] >= sst.lat[1]:
      sstind = sstf.sel(lat=slice(5,-5), lon=slice(190,240)).mean(dim={"lon","lat"})
    else:
      sstind = sstf.sel(lat=slice(-5,5), lon=slice(190,240)).mean(dim={"lon","lat"})
    sstind = sstind.rolling(time=3, center=True).mean().dropna("time")
    # labels asper index 3.4 criteria 1.5 std of the record
    labels=np.zeros_like(sstind.values)
    labels[sstind.values > 1.5*np.std(sstind.values)]=1 # El nino
    labels[sstind.values < -1.5*np.std(sstind.values)]=-1 # La nina
    return labels, sstf


def plot_clusters_iteration(cluster_centers,k):
  clusters = {}
  for i in range(k):
    clusters[i] = ASST.isel(time=0).copy()
    clusters[i].values = cluster_centers[i]
    clusters[i] = clusters[i].unstack('aux')

  fig = plt.figure(figsize=(16, 6),dpi=300,constrained_layout=True)

  for i in range(k):
    lon = clusters[i].lon; lat = clusters[i].lat; data = clusters[i].values
    levels = np.arange(-2,2.25,0.25)
    ax1 = plt.subplot(3,2,i+1)
    im1=ax1.contourf(lon,lat, data.T, levels=levels,cmap='RdBu_r', extend='both')
    ax1.set_title('Cluster #'+str(i+1),fontsize=18, loc='left')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('Longitude', fontsize=16)
    plt.ylabel('Latitude', fontsize=16)
    #Add gridlines

  plt1_ax = plt.gca()
  left, bottom, width, height = plt1_ax.get_position().bounds
  colorbar_axes = fig.add_axes([left + .9, bottom,0.02, height*3])
  cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
  cbar.set_label(r'sea surface temperature anomaly (K)',fontsize=14) #rotation= radianes
  cbar.ax.tick_params(axis='both',labelsize=14)
  fig.tight_layout()
  return fig

def plot_clusters_all_datasets(cluster_centers,k,values,labels,path):

  fig = plt.figure(figsize=(20, 10),dpi=300,constrained_layout=True)
  for j in range(4):
    clusters = {}
    sst_labels, sst = preprocessing(paths[j],subset=[1979])
    asst = sst.stack(aux=('lon','lat'))
    for i in range(k):
      clusters[i] = asst.isel(time=0).copy()
      clusters[i].values = cluster_centers[j]['centers'][i]
      clusters[i] = clusters[i].unstack('aux')

    for i in range(k):
      lon = clusters[i].lon; lat = clusters[i].lat; data = clusters[i].values
      levels = np.arange(-2,2.25,0.25)
      n = j+int(4*i)+1
      ax1 = plt.subplot(7,4,n)
      im1=ax1.contourf(lon,lat, data.T, levels=levels,cmap='RdBu_r', extend='both')
      if labels[j][i] == 1:
        ENSO = 'El Niño'
      elif labels[j][i] == -1:
        ENSO = 'La Niña'
      else:
        ENSO = 'Neutral'
      ax1.set_title('Cluster #'+str(i+1)+' Nino 3.4 $\sigma$: '+str(round(values[j][i],3))+', label:'+ENSO,fontsize=8, loc='left')
      plt.xticks(size=14)
      plt.yticks(size=14)
      plt.xlabel('Longitude', fontsize=12)
      if n == 1:
        plt.ylabel('Latitude', fontsize=12)
      elif n == 5:
        plt.ylabel('Latitude', fontsize=12)
      elif n == 9:
        plt.ylabel('Latitude', fontsize=12)  
      elif n == 13:
        plt.ylabel('Latitude', fontsize=12)
      elif n == 17:
        plt.ylabel('Latitude', fontsize=12)     
      elif n == 21:
        plt.ylabel('Latitude', fontsize=12) 
      #Add gridlines

  plt1_ax = plt.gca()
  left, bottom, width, height = plt1_ax.get_position().bounds
  colorbar_axes = fig.add_axes([left + .3, bottom,0.04, height*8])
  cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
  cbar.set_label(r'sea surface temperature anomaly (K)',fontsize=10) #rotation= radianes
  cbar.ax.tick_params(axis='both',labelsize=10)
  fig.tight_layout()
  return fig


def plot_clusters(cluster_centers,k,values,labels):
  clusters = {}
  for i in range(k):
    clusters[i] = ASST.isel(time=0).copy()
    clusters[i].values = cluster_centers[i]
    clusters[i] = clusters[i].unstack('aux')

  fig = plt.figure(figsize=(16, 6),dpi=300,constrained_layout=True)

  for i in range(k):
    lon = clusters[i].lon; lat = clusters[i].lat; data = clusters[i].values
    levels = np.arange(-2,2.25,0.25)
    ax1 = plt.subplot(3,3,i+1)
    im1=ax1.contourf(lon,lat, data.T, levels=levels,cmap='RdBu_r', extend='both')
    if labels[i] == 1:
      ENSO = 'El Niño'
    elif labels[i] == -1:
      ENSO = 'La Niña'
    else:
      ENSO = 'Neutral'
    ax1.set_title('Cluster #'+str(i+1)+' Nino 3.4 $\sigma$: '+str(round(values[i],3))+', label:'+ENSO,fontsize=12, loc='left')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('Longitude', fontsize=16)
    plt.ylabel('Latitude', fontsize=16)
    #Add gridlines

  plt1_ax = plt.gca()
  left, bottom, width, height = plt1_ax.get_position().bounds
  colorbar_axes = fig.add_axes([left + .9, bottom,0.02, height*3])
  cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
  cbar.set_label(r'sea surface temperature anomaly (K)',fontsize=14) #rotation= radianes
  cbar.ax.tick_params(axis='both',labelsize=14)
  fig.tight_layout()
  return fig

def cluster_vs_month(cluster_labels,data,k=7):
  """
  Function for processing cluster vs months
  Input: cluster label time series (one cluster per month), original data for consistent time index
  Output: dictionary with one annual array per cluster
  """
  time_month = data.month
  clusters = xr.DataArray(cluster_labels,dims = ['month'],name='time_cluster',coords=[time_month])
  k = 7
  cluster_months = {}
  for cluster in range(k):
    cluster_months[cluster] = np.zeros(12)
    for i in range(12):
      cluster_aux = clusters.sel(month=i+1)
      if cluster == 0:
        aux = cluster_aux.where(cluster_aux.data==cluster) +1 
        cluster_months[cluster][i] = aux.sum().values 
      else:
        cluster_months[cluster][i] = cluster_aux.where(cluster_aux.data==cluster).sum().values / cluster

  return cluster_months

def month_vs_freq(cluster_months):
  months = ['J','F','M','A','M','J','J','A','S','O','N','D']
  markers = ['D','D','D','D','D','D','D']
  fig = plt.figure(figsize=(10,5))
  ax = plt.subplot(1,1,1)
  for i in range(7):
    markerline, stemline, baseline, = ax.stem(cluster_months[i], linefmt='white',markerfmt=markers[i],label = 'cluster '+str(i+1))
    plt.setp(stemline, linewidth = 1.25)
    plt.setp(markerline, markersize = 5)
    ax.plot(np.arange(0,12,1),cluster_months[i])
  ax.set_xticks(np.arange(0,12,1))
  ax.set_xticklabels(months)
  ax.legend(bbox_to_anchor=(1.1, 1.05))
  ax.set_xlabel('month')
  ax.set_ylabel('frequency')

def cluster_vs_month_ONDJF(cluster_labels,data,k=7):
  """
  Function for processing cluster vs months
  Input: cluster label time series (one cluster per month), original data for consistent time index
  Output: dictionary with one annual array per cluster
  """
  time_month = data.month
  clusters = xr.DataArray(cluster_labels,dims = ['month'],name='time_cluster',coords=[time_month])
  k = 7
  cluster_months = {}
  for cluster in range(k):
    cluster_months[cluster] = np.zeros(5)
    for i in range(5):
      cluster_aux = clusters.sel(month=i+1)
      if cluster == 0:
        aux = cluster_aux.where(cluster_aux.data==cluster) +1 
        cluster_months[cluster][i] = aux.sum().values 
      else:
        cluster_months[cluster][i] = cluster_aux.where(cluster_aux.data==cluster).sum().values / cluster

  return cluster_months

def month_vs_freq_ONDJF(cluster_months):
  months = ['J','F','O','N','D']
  markers = ['D','D','D','D','D','D','D']
  fig = plt.figure(figsize=(10,5))
  ax = plt.subplot(1,1,1)
  for i in range(7):
    markerline, stemline, baseline, = ax.stem(cluster_months[i], linefmt='white',markerfmt=markers[i],label = 'cluster '+str(i+1))
    plt.setp(stemline, linewidth = 1.25)
    plt.setp(markerline, markersize = 5)
    ax.plot(np.arange(0,5,1),cluster_months[i])
  ax.set_xticks(np.arange(0,5,1))
  ax.set_xticklabels(months)
  ax.legend(bbox_to_anchor=(1.1, 1.05))
  ax.set_xlabel('month')
  ax.set_ylabel('frequency')
