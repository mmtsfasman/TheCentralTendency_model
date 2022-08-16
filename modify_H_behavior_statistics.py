import os
import numpy as np
import pandas as pd

human_tab_mech = np.load('human_RI_tablet_vs_mechanical.npy')
human_tab_soc = np.load('human_RI_tablet_vs_social.npy')
human_mech_soc = np.load('human_RI_mechanical_vs_social.npy')

col_subject = np.tile(np.arange(25).reshape((-1,1)),(3,1))
col_who = np.tile('human', (25*3,1))
col_net = np.tile(-1, (25*3,1))
col_comparison = np.concatenate([np.tile('TM', (25,1)), np.tile('TS', (25,1)), np.tile('MS', (25,1))])
col_value = np.concatenate([human_tab_mech, human_tab_soc, human_mech_soc]).reshape((-1,1))
dataframe = pd.DataFrame(data=np.concatenate([col_subject, col_who, col_net, col_comparison, col_value], axis=1), columns = ['subject', 'who', 'net', 'comparison', 'value'])

std1_TM = np.std(human_tab_mech)
std1_TS = np.std(human_tab_soc)
std1_MS = np.std(human_mech_soc)

model_tab_mech = np.empty((10,), dtype=object)
model_tab_soc = np.empty((10,), dtype=object)
model_mech_soc = np.empty((10,), dtype=object)

std_diff_TM = []
std_diff_TS = []
std_diff_MS = []

for n in range(10):
    model_tab_mech[n] = np.load('results/model_RI_tablet_vs_mechanical_net-' + str(n) + '.npy')
    model_tab_soc[n] = np.load('results/model_RI_tablet_vs_social_net-' + str(n) + '.npy')
    model_mech_soc[n] = np.load('results/model_RI_mechanical_vs_social_net-' + str(n) + '.npy')

    std2_TM = np.std(model_tab_mech[n])
    std2_TS = np.std(model_tab_soc[n])
    std2_MS = np.std(model_mech_soc[n])

    std_diff_TM.append(((std2_TM - std1_TM)/std1_TM)*100)
    std_diff_TS.append(((std2_TS - std1_TS)/std1_TS)*100)
    std_diff_MS.append(((std2_MS - std1_MS)/std1_MS)*100)
    
    col_subject = np.tile(np.arange(25).reshape((-1,1)),(3,1))
    col_who = np.tile('model', (25*3,1))
    col_net = np.tile(n, (25*3,1))
    col_comparison = np.concatenate([np.tile('TM', (25,1)), np.tile('TS', (25,1)), np.tile('MS', (25,1))])
    col_value = np.concatenate([model_tab_mech[n], model_tab_soc[n], model_mech_soc[n]]).reshape((-1,1))
    dataframe = dataframe.append(pd.DataFrame(data=np.concatenate([col_subject, col_who, col_net, col_comparison, col_value],axis=1), columns = ['subject', 'who', 'net', 'comparison', 'value']))

dataframe.to_csv('results/human-model-comparison.csv')

print("Average change of standard deviation in model vs. in human data is " + str(np.mean(np.concatenate([std_diff_TM, std_diff_TS, std_diff_MS]))) + "%")

