import sys
#assert sys.version_info >= (3, 5)
# Common imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import ML_toolbox as mytb
from importlib import reload

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
#Clustering imports
from sklearn.cluster import KMeans

#path='HadISST_sst.nc'  #Add path to SST dataset - 
#path='ersstv5.sst.mnmean.nc'
#path='kaplan.sst.mon.anom.nc'
path='SST_COBE2_1x1.nc'

#Reading and pre-processing SST
sst_labels, sst = mytb.preprocessing(path,subset=[1979])
sst = sst.fillna(0).sel(time=slice('1900','2013'))
asst = sst.stack(aux=('lon','lat'))
time_month = asst.month

sst_labels_before, sst_before = mytb.preprocessing(path,subset=[1900,1979])  #leaving it as 1900 because all data go back to this date
sst_before = sst_before.fillna(0)
asst_before = sst_before.stack(aux=('lon','lat'))
time_month_before = asst_before.month

asst_all=xr.concat([asst_before,asst],dim='time')

lon=sst.lon.values; lat=sst.lat.values
classif=[]
itr=50   #50 iterations
scaler=sklearn.preprocessing.StandardScaler()

for k in range(2,20):
    centers={}
    for ii in range(itr):
        print(k,' clusters', ii+1,' of ',itr)
        kmeans = KMeans(n_clusters=k,random_state=ii)
        cluster=kmeans.fit_predict(asst)
        centers[ii]=kmeans.cluster_centers_

    cc=np.zeros((itr,itr))
    for ii in range(itr):
        data1=np.asarray([centers[ii][i] for i in range(k)]).T
        cc[ii,ii]=np.nan
        for jj in [jj for jj in range(itr) if jj != ii]:
            data2=np.asarray([centers[jj][i] for i in range(k)]).T

            #standardizing the data
            x1=np.mat(scaler.fit_transform(data1))  
            x2=np.mat(scaler.fit_transform(data2))
            ACC=(x2.transpose()*x1)/(x1.shape[0]-1.) #rows -> corr of all x1 with each x2. 

            cc[jj,ii]=ACC.max(1).min()

    print('Calculting index')

    classif.append(np.nansum(cc)/(itr*(itr-1)))    

print(classif)

#Generating random datasets
ave=np.nanmean(asst,0) #spatial mean
cov=np.cov(asst.T)     #spatial covariation

#100 samples with same number of months as asst
art_sst={}
nr=100  #number of random simulations
for ii in range(nr):
    print(ii+1,' of ',nr)
    art_sst[ii]=np.random.multivariate_normal(ave,cov,asst.shape[0])

art_class={}
for kk in range(nr):  #running over each sample
    print('Iteration',kk+1,' of ',nr)
    art_class[kk]=[]
    for k in range(2,20):
        print(' ... Estimating',k,' clusters')
        centers={}
        for ii in range(itr):
            kmeans = KMeans(n_clusters=k,random_state=ii)
            cluster=kmeans.fit_predict(art_sst[kk])
            centers[ii]=kmeans.cluster_centers_

        cc=np.zeros((itr,itr))
        for ii in range(itr):
            data1=np.asarray([centers[ii][i] for i in range(k)]).T
            cc[ii,ii]=np.nan
            for jj in [jj for jj in range(itr) if jj != ii]:
                data2=np.asarray([centers[jj][i] for i in range(k)]).T

                #standardizing the data
                x1=np.mat(scaler.fit_transform(data1))  
                x2=np.mat(scaler.fit_transform(data2))
                ACC=(x2.transpose()*x1)/(x1.shape[0]-1.) #rows -> corr of all x1 with each x2. 

                cc[jj,ii]=ACC.max(1).min()

        print('Calculting index')

        art_class[kk].append(np.nansum(cc)/(itr*(itr-1)))    

output={'ObsClass':classif,'ArtificialClass':art_class}
np.save('ClassifiabilyIndex_COBE2.npy',output,allow_pickle=True)

#Transforming the list into an array
probs=np.asarray([np.asarray(art_classif[ii]) for ii in range(nr)])
#90th, 95th and 97.5th percentile for each k
p90th=np.percentile(probs,90,axis=0)
p95th=np.percentile(probs,95,axis=0)
p97th=np.percentile(probs,97.5,axis=0)

#Plotting

fig=plt.figure(figsize=(3.5,3.5))
plt.subplots_adjust(left=0.2,right=.97,top=.9,bottom=.15)

plt.plot(range(2,20),classif,'-ok')
for ii in range(nr): plt.plot(range(2,20),probs[ii,:],color='lightgray',linewidth=.2,zorder=0)
plt.plot(range(2,20),p90th,'-r',linewidth=1)
plt.plot(range(2,20),p95th,'--r',linewidth=1)
plt.plot(range(2,20),p97th,':r',linewidth=1)

plt.grid(ls=':',lw=.3,color='k')
plt.ylim(.6,1.03); plt.xlim(2,19)
plt.yticks(np.arange(.6,1.03,.05))
plt.ylabel('Classifiability Index',labelpad=1.2)
plt.xlabel('Number of clusters')
plt.title('COBE2')

outf='ClassifiabilityIndex_COBE2.png'
print(outf)
plt.savefig(outf,dpi=150,format="png")

plt.close()
