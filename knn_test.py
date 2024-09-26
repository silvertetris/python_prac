import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings

warnings.filterwarnings('ignore')

data = './diabetes.csv'

df = pd.read_csv(data, header = None)

col_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion',
             'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli']
df.columns=col_names
df.drop('Id', axis=1, inplace=True)
print(df.info)
df = df.apply(pd.to_numeric, errors='coerce')
plt.rcParams['figure.figsize'] = (30,25)
df.plot(kind='hist', bins=10, subplots=True, layout=(5,2), sharex=False, sharey=False)
plt.show()
print(df.dtypes)