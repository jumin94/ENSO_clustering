# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

#My imports
import ML_utilities as mytb

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import sys

#Clustering imports
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#Functions
def correlation_cluster4(Y):
    return np.corrcoef(clusters_all_datasets['Kaplan']['centers'][3],Y)[0][1]

def correlation_cluster2(Y):
    return np.corrcoef(clusters_all_datasets['Kaplan']['centers'][1],Y)[0][1]

def correlation_cluster7(Y):
    return np.corrcoef(clusters_all_datasets['Kaplan']['centers'][6],Y)[0][1]

def correlation_cluster(X,Y):
    corrs = []
    for i in range(len(X)):
        corr  = np.corrcoef(X[i],Y[i])[0][1]
        corrs.append(corr)
    return corrs
   
def final_sorting_center(cluster_analysis,sorted_indices):
    criterio_4 =  correlation_cluster4(cluster_analysis['centers'][sorted_indices[3]])
    criterio_5 =  correlation_cluster4(cluster_analysis['centers'][sorted_indices[4]])
    if (criterio_4 < criterio_5):
        print('keep because correlation of C4 and C5 with Kaplan C4 are: ', criterio_4, criterio_5)
        new_indices = sorted_indices
    else:
        print('change because correlation of C4 and C5 with Kaplan C4 are: ', criterio_4,criterio_5)
        new_indices = sorted_indices.copy()
        for i in range(len(sorted_indices)):
            if i == 3:
                new_indices[i] = sorted_indices[4]
            elif i == 4:
                new_indices[i] = sorted_indices[3]
            else:
                new_indices[i] = sorted_indices[i]
    return new_indices

def final_sorting_LN(cluster_analysis,sorted_indices):
    criterio_1 =  correlation_cluster2(cluster_analysis['centers'][sorted_indices[1]])
    criterio_2 =  correlation_cluster2(cluster_analysis['centers'][sorted_indices[2]])
    if (np.abs(criterio_1) > np.abs(criterio_2)):
        print('keep because correlation of C2 and C3 with Kaplan C4 are: ', np.abs(criterio_1), np.abs(criterio_2))
        new_indices = sorted_indices
    else:
        print('change because correlation of C2 and C3 with Kaplan 4 are: ', np.abs(criterio_1), np.abs(criterio_2))
        new_indices = sorted_indices.copy()
        for i in range(len(sorted_indices)):
            if i == 1:
                new_indices[i] = sorted_indices[2]
            elif i == 2:
                new_indices[i] = sorted_indices[1]
            else:
                new_indices[i] = sorted_indices[i]
    return new_indices

def final_sorting_EN(cluster_analysis,sorted_indices):
    criterio_6 =  correlation_cluster7(cluster_analysis['centers'][sorted_indices[5]])
    criterio_7 =  correlation_cluster7(cluster_analysis['centers'][sorted_indices[6]])
    if (np.abs(criterio_6) < np.abs(criterio_7)):
        print('keep because correlation of C6 and C7 with Kaplan C7 are: ', np.abs(criterio_6), np.abs(criterio_7))
        new_indices = sorted_indices
    else:
        print('change because correlation of C6 and C7 with Kaplan C7 are: ', np.abs(criterio_6), np.abs(criterio_7))
        new_indices = sorted_indices.copy()
        for i in range(len(sorted_indices)):
            if i == 5:
                new_indices[i] = sorted_indices[6]
            elif i == 6:
                new_indices[i] = sorted_indices[5]
            else:
                new_indices[i] = sorted_indices[i]
    return new_indices


def correlate_clusters_detail():
    dataset_list = ['COBE','ERSST','HadISST','Kaplan']
    counter = 0
    clusters = {}
    for dataset in dataset_list:
        clusters[dataset] = {}
        for i in range(8):
            clusters[dataset][i] = xr.open_dataset('/home/julia.mindlin/ENSO_favors/New_classification_eight_clusters/'+dataset+'_cluster_'+str(i+1)+'.nc').sst
            clusters[dataset][i] = clusters[dataset][i].stack(aux=('lon','lat')).values
    correlations =  []
    pairs = ['HadISST-Kaplan','COBE-ERSST','COBE-HadISST','COBE-Kaplan','ERSST-HadISST','ERSST-Kaplan']
    dataset_one = dataset_list[0]
    correlations.append(np.array(correlation_cluster(clusters[dataset_list[2]],clusters[dataset_list[3]])))
    for i,ds in enumerate(['ERSST','HadISST','Kaplan']):	
        correlations.append(np.array(correlation_cluster(clusters[dataset_one],clusters[ds])))
    dataset_two = dataset_list[1]
    for i,ds in enumerate(['HadISST','Kaplan']):
        correlations.append(np.array(correlation_cluster(clusters[dataset_two],clusters[ds])))
    return correlations, pairs

def correlate_clusters():
    dataset_list = ['COBE','ERSST','HadISST','Kaplan']
    counter = 0
    clusters = {}
    for dataset in dataset_list:
        clusters[dataset] = {}
        for i in range(8):
            clusters[dataset][i] = xr.open_dataset('/home/julia.mindlin/ENSO_favors/New_classification_eight_clusters/'+dataset+'_cluster_'+str(i+1)+'.nc').sst
            clusters[dataset][i] = clusters[dataset][i].stack(aux=('lon','lat')).values
    correlations_mean =  []
    correlations_min = []
    correlations_max = []
    dataset_one = dataset_list[0]
    correlations_mean.append(np.mean(np.abs(np.array(correlation_cluster(clusters[dataset_list[2]],clusters[dataset_list[3]])))))
    correlations_min.append(np.min(np.abs(np.array(correlation_cluster(clusters[dataset_list[2]],clusters[dataset_list[3]])))))
    correlations_max.append(np.max(np.abs(np.array(correlation_cluster(clusters[dataset_list[2]],clusters[dataset_list[3]])))))
    for i,ds in enumerate(['ERSST','HadISST','Kaplan']):	
        correlations_mean.append(np.mean(np.abs(np.array(correlation_cluster(clusters[dataset_one],clusters[ds])))))
        correlations_min.append(np.min(np.abs(np.array(correlation_cluster(clusters[dataset_one],clusters[ds])))))
        correlations_max.append(np.max(np.abs(np.array(correlation_cluster(clusters[dataset_one],clusters[ds])))))
    dataset_two = dataset_list[1]
    for i,ds in enumerate(['HadISST','Kaplan']):
        correlations_mean.append(np.mean(np.abs(np.array(correlation_cluster(clusters[dataset_two],clusters[ds])))))
        correlations_min.append(np.min(np.abs(np.array(correlation_cluster(clusters[dataset_two],clusters[ds])))))
        correlations_max.append(np.max(np.abs(np.array(correlation_cluster(clusters[dataset_two],clusters[ds])))))
    return correlations_mean,correlations_min,correlations_max


def time_cluster_agrement(labels_all_datasets):
    dataset_list = ['COBE','ERSST','HadISST','Kaplan']
    agreement = []
    dataset_one = dataset_list[0]
    for i,ds in enumerate(['ERSST','HadISST','Kaplan']):
        dif = labels_all_datasets[dataset_one] - labels_all_datasets[ds]
        agreement.append((dif.size - np.count_nonzero(dif))/dif.size)
    dataset_two = dataset_list[1]
    for i,ds in enumerate(['HadISST','Kaplan']):
        dif = labels_all_datasets[dataset_two] - labels_all_datasets[ds]
        agreement.append((dif.size - np.count_nonzero(dif))/dif.size)
    dif = labels_all_datasets['HadISST'] - labels_all_datasets['Kaplan']
    agreement.append((dif.size - np.count_nonzero(dif))/dif.size)
    return agreement


def plot_clusters(original_grids,cluster_centers,clusters_in_order,values,labels,number): #clusters in order is an array of sorted clusters
    fig, ax = plt.subplots(8,4,figsize=(20, 10),dpi=300,constrained_layout=True)
    dataset_list = ['COBE','ERSST','HadISST','Kaplan']
    counter = 0
    for dataset in dataset_list:
        k = len(clusters_in_order[dataset])
        clusters = {}
        for i in clusters_in_order[dataset]:
            print(i)
            clusters[i] = original_grids[dataset].isel(time=0).copy()
            clusters[i].values = cluster_centers[dataset]['centers'][i]
            clusters[i] = clusters[i].unstack('aux')
        for i in range(k):
            ii = clusters_in_order[dataset][i]
            lon = clusters[i].lon; lat = clusters[i].lat; data = clusters[ii].values
            levels = np.arange(-2,2.25,0.25)
            im1=ax[i,counter].contourf(lon,lat, data.T, levels=levels,cmap='RdBu_r', extend='both')
            if labels[dataset][ii] == 1:
                ENSO = 'El Niño'
            elif labels[dataset][ii] == -1:
                ENSO = 'La Niña'
            else:
                ENSO = 'Neutral'
            (clusters[ii].to_dataset()).to_netcdf('/home/julia.mindlin/ENSO_favors/New_classification_eight_clusters/'+dataset+'_cluster_'+str(i+1)+'.nc')
            plt.xticks(size=14)
            plt.yticks(size=14)
            plt.xlabel('Longitude', fontsize=16)
            plt.ylabel('Latitude', fontsize=16)
            if i == 0:
                ax[i,counter].set_title(dataset+'\n Cluster #'+str(i+1)+' Nino 3.4 $\sigma$: '+str(round(values[dataset][ii],3))+', label:'+ENSO,fontsize=12, loc='left')
            else:
                ax[i,counter].set_title('Cluster #'+str(i+1)+' Nino 3.4 $\sigma$: '+str(round(values[dataset][ii],3))+', label:'+ENSO,fontsize=12, loc='left')
        counter +=1
    #Add gridlines
    cbar = fig.colorbar(im1, ax=ax[:,3])
    cbar.set_label(r'sea surface temperature anomaly (K)',fontsize=14) #rotation= radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    fig.savefig('/home/julia.mindlin/ENSO_favors/New_classification_eight_clusters/All_datasets_Kaplan_grid_1942-2022_'+str(number)+'_years.png')
    plt.close()

def preprocessing_ONI(PathAndFilename):
    '''
        Compute ONI anomalies (written only to start in 1979 having data from before)
    '''
    ds = xr.open_dataset(PathAndFilename)
    if('KAPLAN' in PathAndFilename):
      if ds.lat[0] >= ds.lat[1]:
        sst = ds.sst.sel(lon=slice(140,280), lat=slice(15,-15)) # subset the data
      else:
        sst = ds.sst.sel(lon=slice(140,280), lat=slice(-15,15)) # subset the data
    else:
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
    # preprocessing
    anios = np.arange(1976,2007,5)
    dato_scratch = []
    for anio in anios:
        clim  = sst.sel(time=slice(str(anio+2-15),str(anio+2+15))).groupby('time.month').mean(dim='time')
        dato_scratch.append(sst.sel(time=slice(str(anio),str(anio+4))).groupby('time.month') - clim)
    clim  = sst.sel(time=slice(str(1991),str(2020))).groupby('time.month').mean(dim='time')
    dato_scratch.append(sst.sel(time=slice(str(2011),str(2022))).groupby('time.month') - clim)
    ssta = xr.concat(dato_scratch,dim='time')
    sstd = ssta.rolling(time=3, center=True).mean() # removing trend ssta - ssta - 
    sstf = sstd.dropna('time', how='all') # dropping NaN time steps resulting from rolling mean. 
    return sstf        
            
def preprocessing(PathAndFilename,subset=False):
    '''
        If selecting a specific period, use subset=[yr0,yrf] or subset = [yr0]
    '''
    ds = xr.open_dataset(PathAndFilename)
    if('KAPLAN' in PathAndFilename):
      if ds.lat[0] >= ds.lat[1]:
        sst = ds.sst.sel(lon=slice(140,280), lat=slice(15,-15)) # subset the data
      else:
        sst = ds.sst.sel(lon=slice(140,280), lat=slice(-15,15)) # subset the data
    else:
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

def evaluate_clusters(path,name,y0):
    #sst for the satellite-era period
    if name != 'HadISST':
        sst_labels, sst = mytb.preprocessing(path,subset=[1900,2022])  #leaving it as 1900 because all data go back to this date
    else:
        sst_labels, sst = preprocessing(path,subset=[1900,2022])
#    sst_labels, sst = mytb.preprocessing(path,subset=[1900,2022])
    sst = sst.fillna(0).sel(time=slice(str(y0),'2022'))
    asst = sst.stack(aux=('lon','lat'))
    time_month = asst.month
    #sst for the pre-satellite era period
    if name != 'HadISST':
        sst_labels_before, sst_before = mytb.preprocessing(path,subset=[1900,y0+4])  #leaving it as 1900 because all data go back to this date
    else:
        #sst for the pre-satellite era period
        sst_labels_before, sst_before = preprocessing(path,subset=[1900,y0+4])
#    sst_labels_before, sst_before = mytb.preprocessing(path,subset=[1900,1946])
    sst_before = sst_before[:-1]
    sst_before = sst_before.fillna(0)
    asst_before = sst_before.stack(aux=('lon','lat'))
    time_month_before = asst_before.month
    #Combine both -- all
    asst_all=xr.concat([asst_before,asst],dim='time')
    #Cluster analysis considering 7 clusters
    k=8
    cluster_analysis={}; silhouette={}
    #applying kmean to a subset
    kmeans = KMeans(n_clusters=k,random_state=1234)
    cluster_analysis['satellite era'] = kmeans.fit_predict(asst)
    cluster_analysis['centers'] = kmeans.cluster_centers_
    cluster_analysis['pre-satellite era'] = kmeans.predict(asst_before) #labels for the rest of the time series
    #Concatenating lables for both periods
    cluster_analysis['all']=np.hstack((cluster_analysis['pre-satellite era'],\
                                        cluster_analysis['satellite era']))
    #Labels
    clusters = {}
    labels = []
    values = []
    for i in range(k):
        clusters[i] = asst_all.isel(time=0).copy()
        clusters[i].values = cluster_analysis['centers'][i]
        clusters[i] = clusters[i].unstack('aux')
    ASST = asst_all
    for i in range(k):
        lon = clusters[i].lon; lat = clusters[i].lat; data = clusters[i]
        if lat[0] >= lat[1]:
            sstind = data.sel(lat=slice(5,-5), lon=slice(210,240)).mean(dim={"lon","lat"})
            ASSTind34 = ASST.unstack('aux').sel(lat=slice(5,-5), lon=slice(190,240)).mean(dim={"lon","lat"})
            ASSTind3 = ASST.unstack('aux').sel(lat=slice(5,-5), lon=slice(210,270)).mean(dim={"lon","lat"})
            ASSTind12 = ASST.unstack('aux').sel(lat=slice(0,-10), lon=slice(270,280)).mean(dim={"lon","lat"})
        else:
            sstind = data.sel(lat=slice(-5,5), lon=slice(210,240)).mean(dim={"lon","lat"})
            ASSTind34 = ASST.unstack('aux').sel(lat=slice(-5,5), lon=slice(190,240)).mean(dim={"lon","lat"})
            ASSTind3 = ASST.unstack('aux').sel(lat=slice(-5,5), lon=slice(210,270)).mean(dim={"lon","lat"})
            ASSTind12 = ASST.unstack('aux').sel(lat=slice(-10,0), lon=slice(270,280)).mean(dim={"lon","lat"})
        values.append(sstind.values/np.std(ASSTind34.values)) 
        # labels asper index 3.4 criteria 1.5 std of the record
        if sstind.values > 0.5*np.std(ASSTind34.values):
            labels.append(1)# El nino
        elif sstind.values > 0.5*np.std(ASSTind3.values):
            labels.append(1) # El niño
        elif sstind.values > 0.5*np.std(ASSTind12.values):
            labels.append(1) # El niño
        elif sstind.values < -0.5*np.std(ASSTind34.values):
            labels.append(-1) # La nina
        elif sstind.values < -0.5*np.std(ASSTind3.values):
            labels.append(-1) # La nina
        elif sstind.values < -0.5*np.std(ASSTind12.values):
            labels.append(-1) # La nina
        else:
            labels.append(0)
    return cluster_analysis,values,labels,asst_all

def save_clusters(original_grids,cluster_centers,clusters_in_order,values,labels,number): #clusters in order is an array of sorted clusters
    fig, ax = plt.subplots(8,4,figsize=(20, 10),dpi=300,constrained_layout=True)
    dataset_list = ['COBE','ERSST','HadISST','Kaplan']
    counter = 0
    for dataset in dataset_list:
        k = len(clusters_in_order[dataset])
        clusters = {}
        for i in clusters_in_order[dataset]:
            print(i)
            clusters[i] = original_grids[dataset].isel(time=0).copy()
            clusters[i].values = cluster_centers[dataset]['centers'][i]
            clusters[i] = clusters[i].unstack('aux')
        for i in range(k):
            ii = clusters_in_order[dataset][i]
            lon = clusters[i].lon; lat = clusters[i].lat; data = clusters[ii].values
            clusters[i].to_netcdf('/home/julia.mindlin/ENSO_favors/New_classification_eight_clusters/cluster_'+dataset+'_C'+str(i+1)+'.nc')
            
def evaluate_cajas(path):
    #[160,210,-5,5],[270,280,-10,0],[210,270,-5,5]
    #sst for the satellite-era period
    if name != 'HadISST':
        sst = mytb.preprocessing_ONI2(path)  #leaving it as 1900 because all data go back to this date
    else:
        sst = preprocessing_ONI(path)
#    sst_labels, sst = mytb.preprocessing(path,subset=[1900,2022])
    sst = sst.fillna(0).sel(time=slice('1979-06','2017-05'))
    asst = sst.stack(aux=('lon','lat'))
    asst_all = asst.unstack()
    nino3 = asst_all.sel(lat=slice(-5,5)).sel(lon=slice(210,270)).mean(dim={'lon','lat'})
    nino4 = asst_all.sel(lat=slice(-5,5)).sel(lon=slice(160,210)).mean(dim={'lon','lat'})
    nino12 = asst_all.sel(lat=slice(-10,0)).sel(lon=slice(270,280)).mean(dim={'lon','lat'})
    print(np.min(nino3.values))
    print(np.max(nino3.values))
    box3_condition_nino = (nino3.where((nino3>0.5) & (nino4>-0.5) & (nino4<0.5))/nino3.where((nino3>0.5) & (nino4>-0.5) & (nino4<0.5))).fillna(0)*1
    box4_condition_nino = (nino3.where((nino4>0.5) & (nino3>-0.5) & (nino3<0.5))/nino3.where((nino4>0.5) & (nino3>-0.5) & (nino3<0.5))).fillna(0)*2
    box34_condition_nino = (nino3.where((nino3>0.5) & (nino4>0.5))/nino3.where((nino3>0.5) & (nino4>0.5))).fillna(0)*3
    box3_condition_nina = (nino3.where((nino3<-0.5) & (nino4>-0.5) & (nino4<0.5))/nino3.where((nino3<-0.5) & (nino4>-0.5) & (nino4<0.5))).fillna(0)*(-1)
    box4_condition_nina = (nino3.where((nino4<-0.5) & (nino3>-0.5) & (nino3<0.5))/nino3.where((nino4<-0.5) & (nino3>-0.5) & (nino3<0.5))).fillna(0)*(-2)
    box34_condition_nina = (nino3.where((nino3<-0.5) & (nino4<-0.5))/nino3.where((nino3<-0.5) & (nino4<-0.5))).fillna(0)*(-3)
    box12_condition_nina = (nino12.where(nino12<-0.5)/nino12.where(nino12<-0.5)).fillna(0)*(-1)      
    box12_condition_nino = (nino12.where(nino12>0.5)/nino12.where(nino12>0.5)).fillna(0)*(1)
    conditions_nino = box3_condition_nino + box4_condition_nino + box34_condition_nino
    conditions_nina = box3_condition_nina + box4_condition_nina + box34_condition_nina
    return conditions_nino, conditions_nina, box12_condition_nino, box12_condition_nina

#Open and preprocess data
path_ERSST = '/home/julia.mindlin/ENSO_favors/ENSO_clustering/data/sst.mnmean_ERSST_2022_KAPLAN_grid.nc'
#path_ERSST = '/home/julia.mindlin/datos_sst/data_2022/sst.mnmean_ERSST_2022.nc'
path_HadISST = '/home/julia.mindlin/ENSO_favors/ENSO_clustering/data/HadISST_sst_latest_KAPLAN_grid.nc'
#path = '/home/julia.mindlin/datos_sst/data_2022/HadISST_sst_2022.nc'
path_Kaplan = '/home/julia.mindlin/ENSO_favors/ENSO_clustering/data/sst.mean.anom_Kaplan_2022_KAPLAN_grid.nc'
path_COBE = '/home/julia.mindlin/ENSO_favors/ENSO_clustering/data/sst.mon.mean_COBE_2022_KAPLAN_grid.nc'
#path = '/home/julia.mindlin/datos_sst/data_2022/sst.mon.mean_COBE_2022.nc'

years = [1942]
year_length = [80]
names = ['Kaplan','ERSST','HadISST','COBE']
paths = [path_Kaplan, path_ERSST,path_HadISST,path_COBE]
for y0,number in  zip(years,year_length):
    clusters_all_datasets = {}
    values_all_datasets = {}
    labels_all_datasets = {}
    sorted_all_datasets = {}
    original_grids = {}
    cajas_labels = {}
    box12_condition_labels = {}
    label_caja_nino = {}; label_caja_nina = {}; box12_condition_nino ={}; box12_condition_nina = {}
    for name,path in zip(names,paths):
        cluster_analysis, values, labels,asst_all = evaluate_clusters(path,name,y0)
        label_caja_nino[name], label_caja_nina[name], box12_condition_nino[name], box12_condition_nina[name]  = evaluate_cajas(path)
        cajas_labels[name] = label_caja_nino[name]+label_caja_nina[name]
        box12_condition_labels[name] = box12_condition_nino[name] + box12_condition_nina[name]
    # array 'values' contains the niño3.4 index for each cluster
    # sorted_indices will give the cluster number (assigned by the clustering algorithm) that corresponds to each cluster
        sorted_indices = [i for i, _ in sorted(enumerate(values), key=lambda x: x[1])]
        if name != 'Kaplan':
            print('INDICES :',sorted_indices)
            sorted_indices = final_sorting_center(cluster_analysis,sorted_indices)
            sorted_indices = final_sorting_LN(cluster_analysis,sorted_indices)
            sorted_indices = final_sorting_EN(cluster_analysis,sorted_indices)
            sorted_all_datasets[name] = sorted_indices
            print('NEW INDICES :', sorted_indices)
        else:
            sorted_all_datasets[name] = sorted_indices
        #clustering labels for each month
        clustering_labels = cluster_analysis['all']
    #I generate a map to go from the coldest to warmest - I'm assuming that the similarities between datasets will be enough so that this relationship is maintained
        value_map = {sorted_indices[0]: 1, sorted_indices[1]: 2, sorted_indices[2]: 3,
                sorted_indices[3]: 4, sorted_indices[4]: 5, sorted_indices[5]: 6,
                sorted_indices[6]: 7,sorted_indices[7]: 8}
    # Create a new array with the replaced values using a dictionary lookup
        new_clustering_labels = [value_map.get(x, x) for x in clustering_labels]
        original_grids[name] =  asst_all
        clusters_all_datasets[name] = cluster_analysis
        values_all_datasets[name] = values
        labels_all_datasets[name] = np.array(new_clustering_labels)
        

#ERSST
new_labels_xarray = asst_all.isel(aux=0).copy()
new_labels_xarray.values = labels_all_datasets['ERSST']
new_labels_xarray  = new_labels_xarray.sel(time=slice('1979-06','2017-05'))

years_str = ['1979']*12 
for n in np.arange(1980,2017,1):
    years_str = years_str + [str(n)]*12 
    
labels = pd.DataFrame({'label':np.array(new_labels_xarray.values).astype(float),
                       'year':pd.to_datetime(years_str).strftime('%Y').values,
                       'month':pd.to_datetime(new_labels_xarray.time.values).strftime('%m').values})
cluster_labels = pd.pivot_table(data=labels,index='year',columns='month',values='label')

boxes = pd.DataFrame({'label':np.array(cajas_labels['ERSST'].values).astype(float),
                       'year':pd.to_datetime(years_str).strftime('%Y').values,
                       'month':pd.to_datetime(cajas_labels['ERSST'].time.values).strftime('%m').values})
box_labels = pd.pivot_table(data=boxes,index='year',columns='month',values='label')

cluster_labels = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box_labels = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])

classification = {}
classification['ERSST'] = []
for i,m in enumerate(cluster_labels.index):
    classification['ERSST'].append(cluster_labels.fillna(0).values[i, 6])
    

markers = []
for i,m in enumerate(box12_condition_labels['ERSST'].values):
    if m == 1:
        markers.append(u'△')
    elif m== -1:
        markers.append(u'▽')
    else:
        markers.append(u' ')
        
label_nino = label_caja_nino['ERSST']
labels = pd.DataFrame({'label':np.array(label_nino.values).astype(float),
                       'year':pd.to_datetime(label_nino.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nino.time.values).strftime('%m').values})
box_labels_nino = pd.pivot_table(data=labels,index='year',columns='month',values='label')
label_nina = label_caja_nina['ERSST']
labels = pd.DataFrame({'label':np.array(label_nina.values).astype(float),
                       'year':pd.to_datetime(label_nina.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nina.time.values).strftime('%m').values})
box_labels_nina = pd.pivot_table(data=labels,index='year',columns='month',values='label')
box12_nino = box12_condition_nino['ERSST']
labels = pd.DataFrame({'label':np.array(box12_nino.values).astype(float),
                       'year':pd.to_datetime(box12_nino.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(box12_nino.time.values).strftime('%m').values})
box12_labels_nino = pd.pivot_table(data=labels,index='year',columns='month',values='label')

box12_nina = box12_condition_nina['ERSST']
labes = pd.DataFrame({'label':np.array(label_nina.values).astype(float),
                       'year':pd.to_datetime(label_nina.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nina.time.values).strftime('%m').values})
box12_labels_nina = pd.pivot_table(data=labels,index='year',columns='month',values='label')

box_labels_nino = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box12_labels_nino = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box_labels_nina = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box12_labels_nina = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])

markers_box = []

for i,m in enumerate(years_str):
    if m in ['1982','1991','1997','2015']:
        markers_box.append('Spread EN')
    elif m in ['1986','1987','2002','2006','2009','2018']:
        markers_box.append('C+E EN')
    elif m in ['1990','1994', '2004', '2014','2019']:
        markers_box.append('Central EN')   
    elif m in []:
        markers_box.append('Eastern EN')    
    elif m in ['1999', '2007','2010', '2020']:
        markers_box.append('Spread LN')
    elif m in ['1984', '1988', '1998','2011']:
        markers_box.append('C+E LN')
    elif m in ['1983', '2000','2008']:
        markers_box.append('Central LN')   
    elif m in ['1985', '1995', '1996','2005', '2017']:
        markers_box.append('Eastern LN')   
    else:
        markers_box.append('Neutral')
        

markers = np.array(markers).reshape(38,12)
markers_box = np.array(markers_box).reshape(38,12)
markers_box_ERSST = markers_box

import seaborn  as sns
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#myColors = 'royalblue','lightskyblue','lightblue','yellow','lime','pink','lightcoral','maroon'
myColors = "#3A3A98","#7B7BBA","#BDBDDC","green","yellow","#D5B6B6","#AC6D6D","#832424"
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
fig=plt.figure(figsize=(15,20))
ax = plt.subplot(1,2,1)
cax = inset_axes(ax,
                 width="100%",  # width: 40% of parent_bbox width
                 height="1%",  # height: 10% of parent_bbox height
                 loc='lower left',
                 bbox_to_anchor=(0, 1.02, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
ax = sns.heatmap(cluster_labels.fillna(0),vmin=1, vmax=8,cmap=cmap,linewidths=.5, ax=ax,linecolor='lightgray',cbar_ax=cax,cbar_kws={"orientation": "horizontal"})
for i in range(markers.shape[0]):
    if int(cluster_labels.fillna(0).values[i, 6]) != int(classification['ERSST'][i]):
        text = ax.text(12.3, i+.7, 'C'+str(int(cluster_labels.fillna(0).values[i, 6]))+' (C'+str(int(classification['ERSST'][i]))+')',weight='bold')
    else:
        text = ax.text(12.3, i+.7, 'C'+str(int(cluster_labels.fillna(0).values[i, 6])))  
# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([1.5,2.3,3.1,4,5,5.8,6.7,7.6])
colorbar.set_ticklabels([1,2,3,4,5,6,7,8])
#myColors = 'royalblue','lightskyblue','lightblue','lime','pink','lightcoral','maroon'
myColors = "#3A3A98","#7B7BBA","#BDBDDC","yellow","#D5B6B6","#AC6D6D","#832424"
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
ax = plt.subplot(1,2,2)
cax = inset_axes(ax,
                 width="100%",  # width: 40% of parent_bbox width
                 height="1%",  # height: 10% of parent_bbox height
                 loc='lower left',
                 bbox_to_anchor=(0, 1.02, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
ax = sns.heatmap(box_labels.fillna(0),vmin=-3, vmax=3,cmap=cmap,linewidths=.5, ax=ax,linecolor='lightgray',cbar_ax=cax,cbar_kws={"orientation": "horizontal"})
valfmt = matplotlib.ticker.StrMethodFormatter(markers)
texts = []
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        text = ax.text(j+0.4, i+.7, markers[i, j])
        texts.append(text)
    
for i in range(markers.shape[0]):
    text = ax.text(12.3, i+.7,markers_box[i, 11])
# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks(np.linspace(-2.5,2.5,7))
colorbar.set_ticklabels(['LN3+4','LN4','LN3','Neutral','EN3','EN4','EN3+4'])

new_labels_xarray = asst_all.isel(aux=0).copy()
new_labels_xarray.values = labels_all_datasets['Kaplan']
new_labels_xarray  = new_labels_xarray.sel(time=slice('1979-06','2017-05'))

years_str = ['1979']*12 
for n in np.arange(1980,2017,1):
    years_str = years_str + [str(n)]*12 
    
labels = pd.DataFrame({'label':np.array(new_labels_xarray.values).astype(float),
                       'year':pd.to_datetime(years_str).strftime('%Y').values,
                       'month':pd.to_datetime(new_labels_xarray.time.values).strftime('%m').values})
cluster_labels = pd.pivot_table(data=labels,index='year',columns='month',values='label')
#cluster_labels = labels.pivot_table(index=['year','month'],values='label')

boxes = pd.DataFrame({'label':np.array(cajas_labels['Kaplan'].values).astype(float),
                       'year':pd.to_datetime(years_str).strftime('%Y').values,
                       'month':pd.to_datetime(cajas_labels['Kaplan'].time.values).strftime('%m').values})
box_labels = pd.pivot_table(data=boxes,index='year',columns='month',values='label')

cluster_labels = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box_labels = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])


markers = []
for i,m in enumerate(box12_condition_labels['Kaplan'].values):
    if m == 1:
        markers.append(u'△')
    elif m== -1:
        markers.append(u'▽')
    else:
        markers.append(u' ')
        
label_nino = label_caja_nino['Kaplan']
labels = pd.DataFrame({'label':np.array(label_nino.values).astype(float),
                       'year':pd.to_datetime(label_nino.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nino.time.values).strftime('%m').values})
box_labels_nino = pd.pivot_table(data=labels,index='year',columns='month',values='label')
label_nina = label_caja_nina['Kaplan']
labels = pd.DataFrame({'label':np.array(label_nina.values).astype(float),
                       'year':pd.to_datetime(label_nina.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nina.time.values).strftime('%m').values})
box_labels_nina = pd.pivot_table(data=labels,index='year',columns='month',values='label')
box12_nino = box12_condition_nino['Kaplan']
labels = pd.DataFrame({'label':np.array(box12_nino.values).astype(float),
                       'year':pd.to_datetime(box12_nino.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(box12_nino.time.values).strftime('%m').values})
box12_labels_nino = pd.pivot_table(data=labels,index='year',columns='month',values='label')

box12_nina = box12_condition_nina['Kaplan']
labes = pd.DataFrame({'label':np.array(label_nina.values).astype(float),
                       'year':pd.to_datetime(label_nina.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nina.time.values).strftime('%m').values})
box12_labels_nina = pd.pivot_table(data=labels,index='year',columns='month',values='label')

box_labels_nino = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box12_labels_nino = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box_labels_nina = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box12_labels_nina = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])

markers_box = []

for i,m in enumerate(years_str):
    if m in ['1982','1997','2015']:
        markers_box.append('Spread EN')
    elif m in ['1991','1986','1987','2002','2006','2009','2018']:
        markers_box.append('C+E EN')
    elif m in ['1990','1994', '2004', '2014','2019']:
        markers_box.append('Central EN')   
    elif m in []:
        markers_box.append('Eastern EN')    
    elif m in ['1999', '2007','2010', '2020']:
        markers_box.append('Spread LN')
    elif m in ['1984', '1988', '1998','2011']:
        markers_box.append('C+E LN')
    elif m in ['1983', '2000','2008']:
        markers_box.append('Central LN')   
    elif m in ['1985', '1995', '1996', '2017']:
        markers_box.append('Eastern LN')   
    else:
        markers_box.append('Neutral')

markers = np.array(markers).reshape(38,12)
markers_box = np.array(markers_box).reshape(38,12)

import seaborn  as sns
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#myColors = 'royalblue','lightskyblue','lightblue','yellow','lime','pink','lightcoral','maroon'
myColors = "#3A3A98","#7B7BBA","#BDBDDC","green","yellow","#D5B6B6","#AC6D6D","#832424"
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
fig=plt.figure(figsize=(15,20))
ax = plt.subplot(1,2,1)
cax = inset_axes(ax,
                 width="100%",  # width: 40% of parent_bbox width
                 height="1%",  # height: 10% of parent_bbox height
                 loc='lower left',
                 bbox_to_anchor=(0, 1.02, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
ax = sns.heatmap(cluster_labels.fillna(0),vmin=1, vmax=8,cmap=cmap,linewidths=.5, ax=ax,linecolor='lightgray',cbar_ax=cax,cbar_kws={"orientation": "horizontal"})
for i in range(markers.shape[0]):
    if int(cluster_labels.fillna(0).values[i, 6]) != int(classification['ERSST'][i]):
        text = ax.text(12.3, i+.7, 'C'+str(int(cluster_labels.fillna(0).values[i, 6]))+' (C'+str(int(classification['ERSST'][i]))+')',weight='bold')
    else:
        text = ax.text(12.3, i+.7, 'C'+str(int(cluster_labels.fillna(0).values[i, 6])))  
# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([1.5,2.3,3.1,4,5,5.8,6.7,7.6])
colorbar.set_ticklabels([1,2,3,4,5,6,7,8])
#myColors = 'royalblue','lightskyblue','lightblue','lime','pink','lightcoral','maroon'
myColors = "#3A3A98","#7B7BBA","#BDBDDC","yellow","#D5B6B6","#AC6D6D","#832424"
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
ax = plt.subplot(1,2,2)
cax = inset_axes(ax,
                 width="100%",  # width: 40% of parent_bbox width
                 height="1%",  # height: 10% of parent_bbox height
                 loc='lower left',
                 bbox_to_anchor=(0, 1.02, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
ax = sns.heatmap(box_labels.fillna(0),vmin=-3, vmax=3,cmap=cmap,linewidths=.5, ax=ax,linecolor='lightgray',cbar_ax=cax,cbar_kws={"orientation": "horizontal"})
valfmt = matplotlib.ticker.StrMethodFormatter(markers)
texts = []
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        text = ax.text(j+0.4, i+.7, markers[i, j])
        texts.append(text)
    
for i in range(markers.shape[0]):
    if markers_box[i, 11] != markers_box_ERSST[i, 11]:
        text = ax.text(12.3, i+.7,markers_box[i, 11],weight='bold')
    else:
        text = ax.text(12.3, i+.7,markers_box[i, 11])
# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks(np.linspace(-2.5,2.5,7))
colorbar.set_ticklabels(['LN3+4','LN4','LN3','Neutral','EN3','EN4','EN3+4'])

classification['Kaplan'] = []
for i,m in enumerate(cluster_labels.index):
    classification['Kaplan'].append(cluster_labels.fillna(0).values[i, 6])


new_labels_xarray = asst_all.isel(aux=0).copy()
new_labels_xarray.values = labels_all_datasets['HadISST']
new_labels_xarray  = new_labels_xarray.sel(time=slice('1979-06','2017-05'))

years_str = ['1979']*12 
for n in np.arange(1980,2017,1):
    years_str = years_str + [str(n)]*12 
    
labels = pd.DataFrame({'label':np.array(new_labels_xarray.values).astype(float),
                       'year':pd.to_datetime(years_str).strftime('%Y').values,
                       'month':pd.to_datetime(new_labels_xarray.time.values).strftime('%m').values})
cluster_labels = pd.pivot_table(data=labels,index='year',columns='month',values='label')
#cluster_labels = labels.pivot_table(index=['year','month'],values='label')

boxes = pd.DataFrame({'label':np.array(cajas_labels['HadISST'].values).astype(float),
                       'year':pd.to_datetime(years_str).strftime('%Y').values,
                       'month':pd.to_datetime(cajas_labels['HadISST'].time.values).strftime('%m').values})
box_labels = pd.pivot_table(data=boxes,index='year',columns='month',values='label')

cluster_labels = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box_labels = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])


markers = []
for i,m in enumerate(box12_condition_labels['HadISST'].values):
    if m == 1:
        markers.append(u'△')
    elif m== -1:
        markers.append(u'▽')
    else:
        markers.append(u' ')
        
label_nino = label_caja_nino['HadISST']
labels = pd.DataFrame({'label':np.array(label_nino.values).astype(float),
                       'year':pd.to_datetime(label_nino.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nino.time.values).strftime('%m').values})
box_labels_nino = pd.pivot_table(data=labels,index='year',columns='month',values='label')
label_nina = label_caja_nina['HadISST']
labels = pd.DataFrame({'label':np.array(label_nina.values).astype(float),
                       'year':pd.to_datetime(label_nina.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nina.time.values).strftime('%m').values})
box_labels_nina = pd.pivot_table(data=labels,index='year',columns='month',values='label')
box12_nino = box12_condition_nino['HadISST']
labels = pd.DataFrame({'label':np.array(box12_nino.values).astype(float),
                       'year':pd.to_datetime(box12_nino.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(box12_nino.time.values).strftime('%m').values})
box12_labels_nino = pd.pivot_table(data=labels,index='year',columns='month',values='label')

box12_nina = box12_condition_nina['HadISST']
labes = pd.DataFrame({'label':np.array(label_nina.values).astype(float),
                       'year':pd.to_datetime(label_nina.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nina.time.values).strftime('%m').values})
box12_labels_nina = pd.pivot_table(data=labels,index='year',columns='month',values='label')

box_labels_nino = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box12_labels_nino = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box_labels_nina = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box12_labels_nina = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])

markers_box = []
for i,m in enumerate(years_str):
    if m in ['1982','1997','2015']:
        markers_box.append('Spread EN')
    elif m in ['1991','1986','1987','2002','2009','2018']:
        markers_box.append('C+E EN')
    elif m in ['1990','1994', '2004', '2014','2019']:
        markers_box.append('Central EN')   
    elif m in ['2006']:
        markers_box.append('Eastern EN')    
    elif m in ['1999', '2007','2010', '2020']:
        markers_box.append('Spread LN')
    elif m in ['1984', '1988', '1998','2011']:
        markers_box.append('C+E LN')
    elif m in ['1983', '2000','2008']:
        markers_box.append('Central LN')   
    elif m in ['1985', '1995', '1996','2005', '2017']:
        markers_box.append('Eastern LN')   
    else:
        markers_box.append('Neutral')
        
markers = np.array(markers).reshape(38,12)
markers_box = np.array(markers_box).reshape(38,12)

#myColors = 'royalblue','lightskyblue','lightblue','yellow','lime','pink','lightcoral','maroon'
myColors = "#3A3A98","#7B7BBA","#BDBDDC","green","yellow","#D5B6B6","#AC6D6D","#832424"
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
fig=plt.figure(figsize=(15,20))
ax = plt.subplot(1,2,1)
cax = inset_axes(ax,
                 width="100%",  # width: 40% of parent_bbox width
                 height="1%",  # height: 10% of parent_bbox height
                 loc='lower left',
                 bbox_to_anchor=(0, 1.02, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
ax = sns.heatmap(cluster_labels.fillna(0),vmin=1, vmax=8,cmap=cmap,linewidths=.5, ax=ax,linecolor='lightgray',cbar_ax=cax,cbar_kws={"orientation": "horizontal"})
for i in range(markers.shape[0]):
    if int(cluster_labels.fillna(0).values[i, 6]) != int(classification['ERSST'][i]):
        text = ax.text(12.3, i+.7, 'C'+str(int(cluster_labels.fillna(0).values[i, 6]))+' (C'+str(int(classification['ERSST'][i]))+')',weight='bold')
    else:
        text = ax.text(12.3, i+.7, 'C'+str(int(cluster_labels.fillna(0).values[i, 6])))  
# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([1.5,2.3,3.1,4,5,5.8,6.7,7.6])
colorbar.set_ticklabels([1,2,3,4,5,6,7,8])
#myColors = 'royalblue','lightskyblue','lightblue','lime','pink','lightcoral','maroon'
myColors = "#3A3A98","#7B7BBA","#BDBDDC","yellow","#D5B6B6","#AC6D6D","#832424"
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
ax = plt.subplot(1,2,2)
cax = inset_axes(ax,
                 width="100%",  # width: 40% of parent_bbox width
                 height="1%",  # height: 10% of parent_bbox height
                 loc='lower left',
                 bbox_to_anchor=(0, 1.02, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
ax = sns.heatmap(box_labels.fillna(0),vmin=-3, vmax=3,cmap=cmap,linewidths=.5, ax=ax,linecolor='lightgray',cbar_ax=cax,cbar_kws={"orientation": "horizontal"})
valfmt = matplotlib.ticker.StrMethodFormatter(markers)
texts = []
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        text = ax.text(j+0.4, i+.7, markers[i, j])
        texts.append(text)
    
for i in range(markers.shape[0]):
    if markers_box[i, 11] != markers_box_ERSST[i, 11]:
        text = ax.text(12.3, i+.7,markers_box[i, 11],weight='bold')
    else:
        text = ax.text(12.3, i+.7,markers_box[i, 11])
# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks(np.linspace(-2.5,2.5,7))
colorbar.set_ticklabels(['LN3+4','LN4','LN3','Neutral','EN3','EN4','EN3+4'])

classification['HadISST'] = []
for i,m in enumerate(cluster_labels.index):
    classification['HadISST'].append(cluster_labels.fillna(0).values[i, 6])



new_labels_xarray = asst_all.isel(aux=0).copy()
new_labels_xarray.values = labels_all_datasets['COBE']
new_labels_xarray  = new_labels_xarray.sel(time=slice('1979-06','2017-05'))

years_str = ['1979']*12 
for n in np.arange(1980,2017,1):
    years_str = years_str + [str(n)]*12 
    
labels = pd.DataFrame({'label':np.array(new_labels_xarray.values).astype(float),
                       'year':pd.to_datetime(years_str).strftime('%Y').values,
                       'month':pd.to_datetime(new_labels_xarray.time.values).strftime('%m').values})
cluster_labels = pd.pivot_table(data=labels,index='year',columns='month',values='label')
#cluster_labels = labels.pivot_table(index=['year','month'],values='label')

boxes = pd.DataFrame({'label':np.array(cajas_labels['COBE'].values).astype(float),
                       'year':pd.to_datetime(years_str).strftime('%Y').values,
                       'month':pd.to_datetime(cajas_labels['COBE'].time.values).strftime('%m').values})
box_labels = pd.pivot_table(data=boxes,index='year',columns='month',values='label')

cluster_labels = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box_labels = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])


markers = []
for i,m in enumerate(box12_condition_labels['COBE'].values):
    if m == 1:
        markers.append(u'△')
    elif m== -1:
        markers.append(u'▽')
    else:
        markers.append(u' ')
        
label_nino = label_caja_nino['COBE']
labels = pd.DataFrame({'label':np.array(label_nino.values).astype(float),
                       'year':pd.to_datetime(label_nino.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nino.time.values).strftime('%m').values})
box_labels_nino = pd.pivot_table(data=labels,index='year',columns='month',values='label')
label_nina = label_caja_nina['COBE']
labels = pd.DataFrame({'label':np.array(label_nina.values).astype(float),
                       'year':pd.to_datetime(label_nina.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nina.time.values).strftime('%m').values})
box_labels_nina = pd.pivot_table(data=labels,index='year',columns='month',values='label')
box12_nino = box12_condition_nino['COBE']
labels = pd.DataFrame({'label':np.array(box12_nino.values).astype(float),
                       'year':pd.to_datetime(box12_nino.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(box12_nino.time.values).strftime('%m').values})
box12_labels_nino = pd.pivot_table(data=labels,index='year',columns='month',values='label')

box12_nina = box12_condition_nina['COBE']
labes = pd.DataFrame({'label':np.array(label_nina.values).astype(float),
                       'year':pd.to_datetime(label_nina.time.values).strftime('%Y').values,
                       'month':pd.to_datetime(label_nina.time.values).strftime('%m').values})
box12_labels_nina = pd.pivot_table(data=labels,index='year',columns='month',values='label')

box_labels_nino = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box12_labels_nino = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box_labels_nina = cluster_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])
box12_labels_nina = box_labels.reindex(columns=['06','07','08','09','10','11','12','01','02','03','04','05'])

markers_box = []

for i,m in enumerate(years_str):
    if m in ['1982','1997','2015']:
        markers_box.append('Spread EN')
    elif m in ['1991','1986','1987','2002','2006','2009','2018']:
        markers_box.append('C+E EN')
    elif m in ['1990','1994', '2004', '2014','2019']:
        markers_box.append('Central EN')   
    elif m in []:
        markers_box.append('Eastern EN')    
    elif m in ['2007','2010', '2020']:
        markers_box.append('Spread LN')
    elif m in ['1988', '1998','2011']:
        markers_box.append('C+E LN')
    elif m in ['1983','1999','2000','2008']:
        markers_box.append('Central LN')   
    elif m in ['1984','1985', '1995', '1996','2017']:
        markers_box.append('Eastern LN')   
    else:
        markers_box.append('Neutral')
        
markers = np.array(markers).reshape(38,12)
markers_box = np.array(markers_box).reshape(38,12)

#myColors = 'royalblue','lightskyblue','lightblue','yellow','lime','pink','lightcoral','maroon'
myColors = "#3A3A98","#7B7BBA","#BDBDDC","green","yellow","#D5B6B6","#AC6D6D","#832424"
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
fig=plt.figure(figsize=(15,20))
ax = plt.subplot(1,2,1)
cax = inset_axes(ax,
                 width="100%",  # width: 40% of parent_bbox width
                 height="1%",  # height: 10% of parent_bbox height
                 loc='lower left',
                 bbox_to_anchor=(0, 1.02, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
ax = sns.heatmap(cluster_labels.fillna(0),vmin=1, vmax=8,cmap=cmap,linewidths=.5, ax=ax,linecolor='lightgray',cbar_ax=cax,cbar_kws={"orientation": "horizontal"})
for i in range(markers.shape[0]):
    if int(cluster_labels.fillna(0).values[i, 6]) != int(classification['ERSST'][i]):
        text = ax.text(12.3, i+.7, 'C'+str(int(cluster_labels.fillna(0).values[i, 6]))+' (C'+str(int(classification['ERSST'][i]))+')',weight='bold')
    else:
        text = ax.text(12.3, i+.7, 'C'+str(int(cluster_labels.fillna(0).values[i, 6])))  
# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([1.5,2.3,3.1,4,5,5.8,6.7,7.6])
colorbar.set_ticklabels([1,2,3,4,5,6,7,8])
#myColors = 'royalblue','lightskyblue','lightblue','lime','pink','lightcoral','maroon'
myColors = "#3A3A98","#7B7BBA","#BDBDDC","yellow","#D5B6B6","#AC6D6D","#832424"
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
ax = plt.subplot(1,2,2)
cax = inset_axes(ax,
                 width="100%",  # width: 40% of parent_bbox width
                 height="1%",  # height: 10% of parent_bbox height
                 loc='lower left',
                 bbox_to_anchor=(0, 1.02, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
ax = sns.heatmap(box_labels.fillna(0),vmin=-3, vmax=3,cmap=cmap,linewidths=.5, ax=ax,linecolor='lightgray',cbar_ax=cax,cbar_kws={"orientation": "horizontal"})
valfmt = matplotlib.ticker.StrMethodFormatter(markers)
texts = []
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        text = ax.text(j+0.4, i+.7, markers[i, j])
        texts.append(text)
    
for i in range(markers.shape[0]):
    if markers_box[i, 11] != markers_box_ERSST[i, 11]:
        text = ax.text(12.3, i+.7,markers_box[i, 11],weight='bold')
    else:
        text = ax.text(12.3, i+.7,markers_box[i, 11])
# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks(np.linspace(-2.5,2.5,7))
colorbar.set_ticklabels(['LN3+4','LN4','LN3','Neutral','EN3','EN4','EN3+4'])
             
classification['COBE'] = []
for i,m in enumerate(cluster_labels.index):
    classification['COBE'].append(cluster_labels.fillna(0).values[i, 6])
    

