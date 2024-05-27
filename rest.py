
import pandas as pd 
import numpy as np
from surprise import Dataset, Reader
from surprise import SVD, NMF
from surprise.accuracy import rmse, mse 
from surprise.model_selection import GridSearchCV, train_test_split
from typing import List

from surprise import Dataset
data = Dataset.load_builtin("ml-100k")
param_grid_svd = {
    "lr_all" : [0.1, 0.01, 0.001],
    "reg_all": [0.001, 0.01, 0.0001],
    "n_epochs": [30, 50, 100],
    "n_factors": [100, 150, 300]
}

gs = GridSearchCV(SVD, param_grid=param_grid_svd,
                  measures=["rmse", "mae"])
gs.fit(data)