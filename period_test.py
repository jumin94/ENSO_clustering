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
    fig, ax = plt.subplots(8,4,figsize=(20, 15),dpi=300,constrained_layout=True)
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
            ax[i,counter].tick_params(axis='both', labelsize=16)
            #ax[i,counter].tick_params(axis='y', labelsize=16)
            if counter == 0:
                ax[i,counter].set_ylabel('Latitude', fontsize=18)
            else: 
                'nada'
            if i == 0:
                ax[i,counter].set_title(dataset+'\n Cluster #'+str(i+1)+' Nino 3.4 $\sigma$: '+str(round(values[dataset][ii],3))+', \n label:'+ENSO,fontsize=16, loc='center')
            elif i == 7:
                ax[i,counter].set_xlabel('Longitude', fontsize=18)
            else:
                ax[i,counter].set_title('Cluster #'+str(i+1)+' Nino 3.4 $\sigma$: '+str(round(values[dataset][ii],3))+', \n label:'+ENSO,fontsize=16, loc='center')
        counter +=1
    #Add gridlines
    cbar = fig.colorbar(im1, ax=ax[:,3])
    cbar.set_label(r'sea surface temperature anomaly (K)',fontsize=20) #rotation= radianes
    cbar.ax.tick_params(axis='both',labelsize=16)
    fig.savefig('/home/julia.mindlin/ENSO_favors/New_classification_eight_clusters/All_datasets_Kaplan_grid_1942-2022_'+str(number)+'_years.png')
    plt.close()

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
            
            
            
#Open and preprocess data
path_ERSST = '/home/julia.mindlin/ENSO_favors/ENSO_clustering/data/sst.mnmean_ERSST_2022_KAPLAN_grid.nc'
#path_ERSST = '/home/julia.mindlin/datos_sst/data_2022/sst.mnmean_ERSST_2022.nc'
path_HadISST = '/home/julia.mindlin/ENSO_favors/ENSO_clustering/data/HadISST_sst_latest_KAPLAN_grid.nc'
#path = '/home/julia.mindlin/datos_sst/data_2022/HadISST_sst_2022.nc'
path_Kaplan = '/home/julia.mindlin/ENSO_favors/ENSO_clustering/data/sst.mean.anom_Kaplan_2022_KAPLAN_grid.nc'
path_COBE = '/home/julia.mindlin/ENSO_favors/ENSO_clustering/data/sst.mon.mean_COBE_2022_KAPLAN_grid.nc'
#path = '/home/julia.mindlin/datos_sst/data_2022/sst.mon.mean_COBE_2022.nc'

years =  np.array([1942,1952,1962,1972,1982,1992])
year_length = np.array([80,70,60,50,40,30])
names = ['Kaplan','ERSST','HadISST','COBE']
paths = [path_Kaplan, path_ERSST,path_HadISST,path_COBE]
for y0,number in  zip(years,year_length):
    clusters_all_datasets = {}
    values_all_datasets = {}
    labels_all_datasets = {}
    sorted_all_datasets = {}
    original_grids = {}
    for name,path in zip(names,paths):
        cluster_analysis, values, labels,asst_all = evaluate_clusters(path,name,y0)
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
        value_map = {sorted_indices[0]: 1, sorted_indices[1]: 3, sorted_indices[2]: 2,
                sorted_indices[3]: 4, sorted_indices[4]: 5, sorted_indices[5]: 6,
                sorted_indices[6]: 7,sorted_indices[7]: 8}
    # Create a new array with the replaced values using a dictionary lookup
        new_clustering_labels = [value_map.get(x, x) for x in clustering_labels]
        original_grids[name] =  asst_all
        clusters_all_datasets[name] = cluster_analysis
        values_all_datasets[name] = values
        labels_all_datasets[name] = np.array(new_clustering_labels)
    fig = plot_clusters(original_grids,clusters_all_datasets,sorted_all_datasets,values_all_datasets,labels_all_datasets,number)
    plt.close()
    del fig
    fig = plt.figure()
    for n in names:
        plt.plot(labels_all_datasets[n],label = n)
    plt.legend()
    plt.title(str(number))
    fig.savefig("/home/julia.mindlin/ENSO_favors/New_classification_eight_clusters/time_evolution_"+str(number)+".png")
    f = open("/home/julia.mindlin/ENSO_favors/New_classification_eight_clusters/All_datasets_spatial.txt", "a")
    spatial_corr_mean, spatial_corr_min, spatial_corr_max  = correlate_clusters()
    f.write(str(number)+' years: minimum of mean correlations'+str(np.min(spatial_corr_mean))+' minimum of minimum correlations: '+str(np.min(spatial_corr_min))+' minimum of maximum correlations: '+str(np.min(spatial_corr_max))+"\n ") #[0])+', '+str(spatial_corr[1])+', '+str(spatial_corr[2])+', '+str(spatial_corr[3])+',  '+str(spatial_corr[4])+', '+str(spatial_corr[5]))
    f.close()
    spatial_corr_all, pairs = correlate_clusters_detail()
    f = open("/home/julia.mindlin/ENSO_favors/New_classification_eight_clusters/All_datasets_spatial_all.txt", "a")
    f.write(str(number)+' years: \n')
    for kk,pair in enumerate(pairs):
        print(number,pair,spatial_corr_all[kk])
        f.write(pair+': C1_corr: ' +str(spatial_corr_all[kk][0])+' C2_corr: '+str(spatial_corr_all[kk][1])+' C3_corr: '+str(spatial_corr_all[kk][2])+' C4_corr: '+str(spatial_corr_all[kk][3])+' C5_corr: '+str(spatial_corr_all[kk][4])+' C6_corr: '+str(spatial_corr_all[kk][5])+' C7_corr: '+str(spatial_corr_all[kk][6])+' C8_corr: '+str(spatial_corr_all[kk][7])+' \n')
    f.close()
    agreement = np.array(time_cluster_agrement(labels_all_datasets))
    f = open("/home/julia.mindlin/ENSO_favors/New_classification_eight_clusters/All_datasets_temporal.txt", "a")
    f.write(str(number)+' years: minimum agreement percentage: '+str(np.min(agreement))+', mean agreement percentage: \n '+str(np.mean(agreement))+', max agreement percentage: \n '+str(np.max(agreement)))
    f.close()
    f = open("/home/julia.mindlin/ENSO_favors/New_classification_eight_clusters/All_datasets_temporal_max.txt", "a")
    f.write(str(number)+' years: maxmum agreement percentage: '+str(np.max(agreement))+'\n ')
    f.close()
