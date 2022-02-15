import numpy as np
import flowkit as fk
import pandas as pd
import ntpath

#TO-DO
#simplify to 10000 cells per fcs file/patient (only raw)
#keep track of which file each cell came from-for loop or parse



#function to create matrix
def create_matrix(filePath):
    sample = fk.Sample(filePath)
    #print("Information about Sample: ", sample)
    dm = sample._get_raw_events()
    CNames = np.array(sample.pns_labels)
    #select markers of interest
    #toKeep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]
    toKeep = [2,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,49]
    dm_trans= np.arcsinh(1./5 * dm[:,toKeep])
    CNames = CNames[toKeep]
    df_nice = pd.DataFrame(dm_trans,columns=CNames)
    df_nice = df_nice.head(10)
    df_nice.insert(0,"File Name", [ntpath.basename(filePath) for i in range(10)], True)
    #df_nice.insert(0, "Index", [i for i in range(10)], True)
    return df_nice
    #df_columns = pd.DataFrame(df_nice.columns)
    #print("Columns Info: ", df_columns)

#merge all raw files
end = pd.concat([create_matrix('FlowRepository_FR-FCM-ZYJP_files/BC12_all_cells_raw.fcs'), create_matrix('FlowRepository_FR-FCM-ZYJP_files/BC13_all_cells_raw.fcs')], sort = False)
end = pd.concat([end,create_matrix('FlowRepository_FR-FCM-ZYJP_files/BC14_all_cells_raw.fcs')],sort = False)
print(end)

#first raw file
#print(create_matrix('FlowRepository_FR-FCM-ZYJP_files/BC12_all_cells_raw.fcs'))
#second raw file
#print(create_matrix('FlowRepository_FR-FCM-ZYJP_files/BC13_all_cells_raw.fcs'))

# #third raw file
# create_matrix('FlowRepository_FR-FCM-ZYJP_files/BC14_all_cells_raw.fcs')






