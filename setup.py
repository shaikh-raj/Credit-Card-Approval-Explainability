# Setup for importing necessary packages

# Install dependencies
!pip install lime
!pip install shap

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

# For LIME and SHAP explainability
from lime.lime_tabular import LimeTabularExplainer
from lime import submodular_pick
import shap

# To ignore warnings
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
