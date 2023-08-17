### This file evaluates a synthetic dataset for 30 year period to then compare bootstrap sample against this noise. 

# Common imports
import numpy as np
import xarray as xr
from importlib import reload
import pandas as pd
import random 
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
#Clustering imports
from sklearn.cluster import KMeans
#Own functions imports
import ML_utilities as mytb
#Parallel processing imports
import concurrent.futures
import multiprocessing

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


#Classifiability index for each number of clusters and synthetic data 
def worker4(process):  
    class_data = []
    nr=5  #number of random simulations
    print('start worker '+str(process))
    for kk in range(nr):  #running over each sample
        print('Iteration',kk+1,' of ',nr)
        iteration = nr*process+kk
        print(iteration)
        random_sample = [random.randint(0, len(asst.time)-1) for _ in range(50*12)]
        sample_data = asst.isel(time=random_sample)
        output = apply_operation_in_parallel(sample_data,100,20) #this 100 is iternation number (not years!)
        class_data.append(output)
    return class_data


#Open data
path='/home/julia/Desktop/ENSO_flavors_WAF/stationarity_test/data/sst.mnmean_ERSST_2022_KAPLAN_grid.nc'
#Reading and pre-processing SST
sst_labels, sst = mytb.preprocessing(path,subset=[1900,2022])  #leaving it as 1900 because all data go back to this date
sst = sst.fillna(0)
asst = sst.stack(aux=('lon','lat'))
time_mont = asst.month
lon=sst.lon.values; lat=sst.lat.values

#Evaluate classifiability
print('abro ruido')
artificial_class_df = pd.read_csv('/home/julia/Desktop/ENSO_flavors_WAF/stationarity_test/ERSST/50_year_noise_classification_dictionary_ERSST.csv')

art_class = []
for key in artificial_class_df.keys():
    for ii in range(20):
        art_class.append(artificial_class_df[key][ii])
        
class_data={}
with multiprocessing.Pool(processes=5) as pool:
    class_data = pool.map(worker4, range(5))
    
class_data_dict = {}
for i, result in enumerate(class_data):
    class_data_dict[f"Worker {i}"] = result

(pd.DataFrame(class_data_dict)).to_csv('/home/julia/Desktop/ENSO_flavors_WAF/stationarity_test/ERSST/50_year_classification_data_dictionary_4of4.csv')

