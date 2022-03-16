import numpy as np
import flowkit as fk
import pandas as pd
import ntpath
import sklearn
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
# %matplotlib inline

#function to create matrix
def create_matrix(filePath):
    sample = fk.Sample(filePath)
    dm = sample._get_raw_events()
    CNames = np.array(sample.pns_labels)

    #select markers of interest
    #toKeep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]
    toKeep = [2,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,49]

    dm_trans= np.arcsinh(1./5 * dm[:,toKeep])
    CNames = CNames[toKeep]
    df_nice = pd.DataFrame(dm_trans,columns=CNames)
    df_nice = df_nice.head(1000)
    #change to 1000
    df_nice.insert(0,"File Name", [ntpath.basename(filePath) for i in range(1000)], True)
    #df_nice.insert(0, "Index", [i for i in range(10)], True)
    return df_nice
    

#merge all raw files
end = pd.concat([create_matrix('FlowRepository_FR-FCM-ZYJP_files/BC12_all_cells_raw.fcs'), create_matrix('FlowRepository_FR-FCM-ZYJP_files/BC13_all_cells_raw.fcs')], sort = False)
end = pd.concat([end,create_matrix('FlowRepository_FR-FCM-ZYJP_files/BC14_all_cells_raw.fcs')],sort = False)
#print(end)
colsOnly = np.array(end.columns[1:])
print("\n",colsOnly)


x = end.loc[:, colsOnly].values
y = end.loc[:, ['File Name']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
print("Just components?", principalDf)
#print(end[['File Name']])

fileNameOnly = end[['File Name']].set_index([pd.Index( i for i in range(3000))])
#print(fileNameOnly)

finalDf = pd.concat([principalDf, fileNameOnly], axis = 1)

#______________________________________________________________________________________________

#Plotting first PCA 
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['BC12_all_cells_raw.fcs', 'BC13_all_cells_raw.fcs', 'BC14_all_cells_raw.fcs']
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
     indicesToKeep = finalDf['File Name'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

#Plotting second PCA

# indicesToKeep = np.full((3000), True)

# for col in colsOnly:
#   t = end[[col]].to_numpy()
#   plt.scatter(finalDf.loc[indicesToKeep,'principal component 1']
#             , finalDf.loc[indicesToKeep,'principal component 2']
#             ,c = t
#             ,cmap= "cool"
#             , s = 50)
#   plt.gca().update(dict(title= '2 Component PCA by Feature: '+ col, xlabel='Principal Component 1', ylabel='Principal Component 2'))
#   plt.grid()
#   plt.colorbar(label= col, orientation= "vertical")
#   plt.show()



#for loop over features(0 to num columns)
  #continuous color bar for each markers, such as CD45
#clustering sample (k-means clustering) - partition cells into populations based on the clusters, each cells can only belong to one cluster
#Cd123 = dendrite 
#clustering-unsupervised, each cluster has set of similar sets
#room 305


#K Means Clustering

# model = KMeans(n_clusters = 10)
# label = model.fit_predict(principalDf.iloc[:,:3])
# u_labels = np.unique(label)
 
# for i in u_labels:
#     plt.scatter(principalDf[label == i]['principal component 1'] , principalDf[label == i]['principal component 2'] , label = i)
# plt.legend()
# plt.xlabel('Principal component 1')
# plt.ylabel('Principal component 2')
# plt.title('K Means Clustering with 2 Component PCA')
# plt.show()

#Spectral Clustering
# model_2 = SpectralClustering(n_clusters = 10)
# label = model_2.fit_predict(principalDf.iloc[:,:3])
# u_labels = np.unique(label)
 
# for i in u_labels:
#     plt.scatter(principalDf[label == i]['principal component 1'] , principalDf[label == i]['principal component 2'] , label = i)
# plt.legend()
# plt.xlabel('Principal component 1')
# plt.ylabel('Principal component 2')
# plt.title('Spectral Clustering with 2 Component PCA')
# plt.show()

#Agglomerative
# model_3 = AgglomerativeClustering(n_clusters= 10)
# label = model_3.fit_predict(principalDf.iloc[:,:3])
# u_labels = np.unique(label)

# for i in u_labels:
#   plt.scatter(principalDf[label == i]['principal component 1'] , principalDf[label == i]['principal component 2'] , label = i)
# plt.legend()
# plt.xlabel('Principal component 1')
# plt.ylabel('Principal component 2')
# plt.title('Agglomerative Clustering with 2 Component PCA')
# plt.show()







