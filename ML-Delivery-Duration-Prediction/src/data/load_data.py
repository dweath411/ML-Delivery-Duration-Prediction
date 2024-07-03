# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# libraries will besides numpy, pandas seaborn, and matplotlib will be imported as needed and moved here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# datetime dependency
from datetime import datetime

# VIF dependency
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Random Forest dependencies
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Scaling and Dimensionality dependencies
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# MSE dependency
from sklearn.metrics import mean_squared_error

# Regression dependencies
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import linear_model

# Deep Learning dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(42)

# load in data as pandas df (generic, I know)
df = pd.read_csv('historical_data.csv')

# find info on the data, ensure the data type is correct for each column.
df.info()