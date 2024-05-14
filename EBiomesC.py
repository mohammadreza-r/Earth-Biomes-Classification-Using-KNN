# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier as KNCf
from colorama import Fore,Back
# Number of Months
m={"January":1,"February":2,"March":3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
# Collected Data Of Biomes
D_array = [[1,15,-27],[2,12,-26],[3,14,-22],[4,8,-12],[5,15,0],[6,19,10],[7,40,14],[8,42,11],[9,30,4],[10,33,-7],[11,22,-20],[12,16,-25],[1,6,14],[2,5,15],[3,4,17],[4,2,22],[5,1,25],[6,0,27],[7,0,28],[8,0,28],[9,0,27],[10,1,24],[11,3,19],[12,6,15],[1,44,-7],[2,32,-7],[3,36,-3],[4,34,3],[5,35,10],[6,54,15],[7,79,17],[8,81,15],[9,64,9],[10,64,4],[11,60,-1],[12,52,-5],[1,41,6],[2,33,8],[3,25,10],[4,45,12],[5,52,16],[6,25,21],[7,15,25],[8,11,24],[9,27,20],[10,47,15],[11,55,10],[12,54,7],[1,156,20],[2,107,19],[3,100,18],[4,55,15],[5,15,12],[6,7,9],[7,4,9],[8,8,12],[9,30,15],[10,76,17],[11,120,18],[12,132,19],[1,46,10],[2,56,12],[3,49,16],[4,69,20],[5,122,24],[6,96,27],[7,49,29],[8,51,29],[9,88,26],[10,87,21],[11,65,15],[12,50,11],[1,288,27],[2,289,27],[3,308,27],[4,310,27],[5,245,27],[6,122,27],[7,92,27],[8,68,28],[9,88,28],[10,122,28],[11,174,28],[12,217,27]]
trg_nms = ['Tundra','Tundra','Tundra','Tundra','Tundra','Tundra','Tundra','Tundra','Tundra','Tundra','Tundra','Tundra','Desert','Desert','Desert','Desert','Desert','Desert','Desert','Desert','Desert','Desert','Desert','Desert','Coniferous Forest','Coniferous Forest','Coniferous Forest','Coniferous Forest','Coniferous Forest','Coniferous Forest','Coniferous Forest','Coniferous Forest','Coniferous Forest','Coniferous Forest','Coniferous Forest','Coniferous Forest','Temperate Deciduous Forest','Temperate Deciduous Forest','Temperate Deciduous Forest','Temperate Deciduous Forest','Temperate Deciduous Forest','Temperate Deciduous Forest','Temperate Deciduous Forest','Temperate Deciduous Forest','Temperate Deciduous Forest','Temperate Deciduous Forest','Temperate Deciduous Forest','Temperate Deciduous Forest','Shrubland','Shrubland','Shrubland','Shrubland','Shrubland','Shrubland','Shrubland','Shrubland','Shrubland','Shrubland','Shrubland','Shrubland','Grassland','Grassland','Grassland','Grassland','Grassland','Grassland','Grassland','Grassland','Grassland','Grassland','Grassland','Grassland','Rainforest','Rainforest','Rainforest','Rainforest','Rainforest','Rainforest','Rainforest','Rainforest','Rainforest','Rainforest','Rainforest','Rainforest']
cols = ['Month number','Average Monthly Precipitation (mm)','Average Monthly Temperature (°C)']

### Data Collected From NASA ###
### Average(1970-2000) ###

# Make And Edit The Dataframe
Bio = pd.DataFrame(D_array,columns=cols)
Bio['Biomes'] = trg_nms
Bio=Bio.astype({'Biomes':'object'})
Bio_numeric = Bio.drop('Biomes',axis=1)

# KNN Model With Dynamic Value Input
knn=KNCf(n_neighbors=10,metric='minkowski',p=1)
x = np.array(D_array)
y = np.array(trg_nms)
knn.fit(x,y)
month_code = int(input("Enter The Month Number: "))
A_M_P = int(input("Enter The Average Monthly Precipitation (mm): "))
A_M_T = int(input("Enter The Average Monthly Temperature (°C): "))
xx=np.array([[month_code,A_M_P,A_M_T]])
yy=knn.predict(xx)
print(Fore.YELLOW + '\n----------------------------------------------------------------------------' + Fore.RESET)
print('\n' + Fore.CYAN + 'The Information You Gave Contains With ' + Fore.BLACK + Back.WHITE + yy[0] + Fore.CYAN + Back.RESET + ' Biome!')
