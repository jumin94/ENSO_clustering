### This file evaluates a synthetic dataset for 30 year period to then compare bootstrap sample against this noise. 

# Common imports
import numpy as np
import xarray as xr
from importlib import reload
import pandas as pd
import random
#ML
import sklearn
assert sklearn.__version__ >= "0.20"
#Clustering imports
from sklearn.cluster import KMeans
#Own functions imports
import ML_utilities as mytb
#Parallel processing imports
import concurrent.futures

