### This file evaluates a synthetic dataset for n year period to then compare bootstrap sample against this noise. 

#Must select path for data, dataset, change name of output and number of years

# Common imports
import numpy as np
import xarray as xr
from importlib import reload
import pandas as pd
import random 
import sklearn
assert sklearn.__version__ >= "0.20"
#Clustering imports
from sklearn.cluster import KMeans
#Own functions imports
import ML_utilities as mytb
#Parallel processing imports
import concurrent.futures

year_num = 60
data_file_path = '/home/users/tabu/ENSO_clustering/data/HadISST_sst_latest_KAPLAN_grid.nc'
output_file_path = '/home/users/tabu/ENSO_clustering/'+str(year_num)+'_year_noise_classification_dictionary_COBE.csv' 

#FUNCTIONS--------------------------------------------------------------------
def apply_operation_in_parallel(data, itr, k):
    """Applies operation to data based on an integer number (number of clusters) and runs iterations in parallel."""
    def evaluate_clusters(k):
        """Evaluates clusters for k centroids"""
        scaler=sklearn.preprocessing.StandardScaler()
        centers={}
        for ii in range(itr):
            #print(k,' clusters', ii+1,' of ',itr)
            kmeans = KMeans(n_clusters=k,random_state=ii)
            cluster=kmeans.fit_predict(data)
            centers[ii]=kmeans.cluster_centers_

        cc=np.zeros((itr,itr)) #correlation matrix
        for ii in range(itr):
            data1=np.asarray([centers[ii][i] for i in range(k)]).T #iterate over clusters 
            cc[ii,ii]=np.nan
            for jj in [jj for jj in range(itr) if jj != ii]:
                data2=np.asarray([centers[jj][i] for i in range(k)]).T

                #standardizing the data
                x1=np.mat(scaler.fit_transform(data1))  
                x2=np.mat(scaler.fit_transform(data2))
                ACC=(x2.transpose()*x1)/(x1.shape[0]-1.) #rows -> corr of all x1 with each x2. 

                cc[jj,ii]=ACC.max(1).min()

        print('Calculting index')

        classif_k = np.nansum(cc)/(itr*(itr-1)) #out of the spatial correlation and iteration number evaluates the classifiability index
        return classif_k
    
    # Create a list of integer numbers from 1 to num
    ks = list(range(1, k+1))
    
    # Use a ThreadPoolExecutor to run the operation in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the operation for each integer number in nums
        futures = [executor.submit(evaluate_clusters, n) for n in ks]
        
        # Wait for all futures to complete and return the results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return results #the result is the complete set of classifiability index for each number of clusters 

#Synthetic data generation
def worker1(process):  
    ast_sst = []
    nr=20  #number of random simulations
    print('start worker '+str(process))
    for ii in range(nr):
        print(ii+1,' of ',nr)
        art_sst[ii]=np.random.multivariate_normal(ave,cov,sample_data.shape[0])
    return art_sst

#Classifiability index for each number of clusters and synthetic data 
def worker2(process):  
    art_class = []
    nr=1  #number of random simulations
    print('start worker '+str(process))
    for kk in range(nr):  #running over each sample
        print('Iteration',process)
        output = apply_operation_in_parallel(art_sst[process],100,20)
        art_class.append(output)
    return art_class


#Open data and pre-processing SST
sst_labels, sst = mytb.preprocessing(data_file_path,subset=[1900,2022])  #leaving it as 1900 because all data go back to this date
sst = sst.fillna(0)
asst = sst.stack(aux=('lon','lat'))
time_mont = asst.month
lon=sst.lon.values; lat=sst.lat.values

#Evaluate covariance and mean of data for subsamples of 50 years and then take the mean -estimate of representative covariance
cov_matrix = np.zeros([50,168,168])
ave_matrix = np.zeros([50,168])
for i in range(50):
    random_sample = [random.randint(0, len(asst.time)-1) for _ in range(year_num*12)] #50 is number of years
    sample_data = asst.isel(time=random_sample)
    #Generating covariance
    ave_matrix[i,:]   = np.nanmean(sample_data,0) #spatial mean
    cov_matrix[i,:,:] = np.cov(sample_data.T)     #spatial covariation
    
ave = np.mean(ave_matrix,axis=0)
cov = np.mean(cov_matrix,axis=0)

import multiprocessing
#Generation of synthetic data in parallel
print('simulated samples')
art_sst={}
with multiprocessing.Pool(processes=5) as pool:
    art_sst = pool.map(worker1, range(5))

art_sst_dict = {}
for i, result in enumerate(art_sst):
    art_sst_dict["Worker "+str(i)] = result
        
#Reacomodate all in one list
art_sst = []
for key in art_sst_dict.keys():
    for ii in range(len(art_sst_dict[key])):
        art_sst.append(art_sst_dict[key][ii])

del art_sst_dict

#Evaluate the classifiability index for each iteration in parallel
print('clusters of simulated samples')
art_class={}
with multiprocessing.Pool(processes=5) as pool:
    art_class = pool.map(worker2, range(5))
    
art_class_dict = {}
for i, result in enumerate(art_class):
    art_class_dict["Worker "+str(i)] = result
      
count = 4
print('saved ',count)
for jj in range(19):
    art_class={}
    with multiprocessing.Pool(processes=5) as pool:
        art_class = pool.map(worker2, range(5*(jj+1),5*(jj+2)))

    for i, result in enumerate(art_class):
        count = count+1
        print('saving ',count)
        art_class_dict["Worker "+str(count)] = result
        
#Save dictionary
(pd.DataFrame(art_class_dict)).to_csv(output_file_path)
