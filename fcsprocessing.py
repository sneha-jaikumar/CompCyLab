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
#print("Just components?", principalDf)
#print(end[['File Name']])

fileNameOnly = end[['File Name']].set_index([pd.Index( i for i in range(3000))])
#print(fileNameOnly)
finalDf = pd.concat([principalDf, fileNameOnly], axis = 1)

#______________________________________________________________________________________________

# #Plotting first PCA 
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = ['BC12_all_cells_raw.fcs', 'BC13_all_cells_raw.fcs', 'BC14_all_cells_raw.fcs']
# colors = ['r', 'g', 'b']

# for target, color in zip(targets,colors):
#      indicesToKeep = finalDf['File Name'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
# plt.show()

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



#clustering sample (k-means clustering) - partition cells into populations based on the clusters, each cells can only belong to one cluster
#Cd123 = dendrite 
#clustering-unsupervised, each cluster has set of similar sets


#K Means Clustering
model = KMeans(n_clusters = 10)
label = model.fit_predict(finalDf.iloc[:,:2])
u_labels = np.unique(label)
 
# for i in u_labels:
#     plt.scatter(principalDf[label == i]['principal component 1'] , principalDf[label == i]['principal component 2'] , label = i)
# plt.legend()
# plt.xlabel('Principal component 1')
# plt.ylabel('Principal component 2')
# plt.title('K Means Clustering with 2 Component PCA')
# plt.show()

print("labels", u_labels)
temp_vectorBC12 =  [0] * 10
temp_vectorBC13 =  [0] * 10
temp_vectorBC14 =  [0] * 10
for i in u_labels:
  print(len(finalDf[label == i]))
  temp_vectorBC12[i] += sum(finalDf[label == i]['File Name'] == 'BC12_all_cells_raw.fcs')
  temp_vectorBC13[i] += sum(finalDf[label == i]['File Name'] == 'BC13_all_cells_raw.fcs')
  temp_vectorBC14[i] += sum(finalDf[label == i]['File Name'] == 'BC14_all_cells_raw.fcs')

print("BC12:",temp_vectorBC12)
print("BC13",temp_vectorBC13)
print("BC14",temp_vectorBC14)

BC12_vector = np.array(temp_vectorBC12)
BC13_vector = np.array(temp_vectorBC13)
BC14_vector = np.array(temp_vectorBC14)

freq_features_matrix = np.array([BC12_vector, BC13_vector, BC14_vector])
print(freq_features_matrix)


# #Spectral Clustering
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

# #Agglomerative Clustering
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



#next steps
#so for K means clustering, you have 10 clusters
# for file in filetype: #BC12, BC13, BC14
#   while file is BC12, 


# cluster 1:         cluster 2:          cluster 3:
#   34 cells from BC12     455 cells from 




