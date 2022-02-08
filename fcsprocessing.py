import numpy as np
import flowkit as fk
import pandas as pd

fcs_path =  'FlowRepository_FR-FCM-ZYJP_files/BC12_all_cells_raw.fcs'

sample = fk.Sample(fcs_path)


print(sample)

dm = sample._get_raw_events()

CNames = np.array(sample.pns_labels)

toKeep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]

##################################################################
#yfinally create your data matrix for further analysis
##################################################################

#now we will transform the data and select our columns/markers of interest
dm_trans= np.arcsinh(1./5 * dm)
CNames = CNames[toKeep]

#you could have stopped there, but we can make a 'nice' dataframe
df_nice = pd.DataFrame(dm_trans,columns=CNames)
print(df_nice)

df_columns = pd.DataFrame(df_nice.columns)

print(df_columns)

