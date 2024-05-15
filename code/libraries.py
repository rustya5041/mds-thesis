# general modules
from warnings import filterwarnings
from glob import glob
import json
import pandas as pd
import numpy as np
from ast import literal_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader

# vis
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from torch_geometric.nn import GCNConv

# evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

pd.set_option('display.max_columns', 199)
filterwarnings('ignore')