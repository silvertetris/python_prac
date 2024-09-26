import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings

warnings.filterwarnings('ignore')

data = './diabetes.csv'

df = pd.read_csv(data, header = None)

print(df.shape, df.head)
print(df.columns)