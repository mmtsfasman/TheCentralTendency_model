# Calculate summary across networks
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
num_timesteps = 22

train_H = 1
test_H = 1

folders = [
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2020-12-18_17-01_0997494/' + str(train_H) + '/evaluation',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-21_13-38_0701178/' + str(train_H) + '/evaluation',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-21_13-46_0242292/' + str(train_H) + '/evaluation',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-21_13-47_0425773/' + str(train_H) + '/evaluation',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-21_13-47_0470388/' + str(train_H) + '/evaluation',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-21_13-48_0583639/' + str(train_H) + '/evaluation',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-22_10-26_0684351/' + str(train_H) + '/evaluation',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-22_10-27_0506050/' + str(train_H) + '/evaluation',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-22_10-27_0591835/' + str(train_H) + '/evaluation',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-22_10-27_0679615/' + str(train_H) + '/evaluation',
        ]

evaluation_folder = 'results/network_summary/'
pathlib.Path(evaluation_folder).mkdir(parents=True, exist_ok=True)

######################################
# SUMMARY ACROSS NETWORK PERFORMANCE #
######################################

human_RI = np.empty((len(folders),),dtype=object)
model_RI = np.empty((len(folders),),dtype=object)
num_conditions = 3
num_participants = 25
for i, f in enumerate(folders):
    human_RI[i] = np.load(os.path.join(f, 'H-1_humanRI.npy'))
    model_RI[i] = np.load(os.path.join(f, 'H-1_modelRI.npy'))

colors = ['r', '#f3a712', 'b']
markers = ['^', 's', 'd']
fig = plt.figure(figsize=(8,7))
parameters = {'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'axes.labelsize': 24,
          'axes.titlesize': 24}
plt.rcParams.update(parameters)
xmin = np.min([np.concatenate(model_RI), np.concatenate(human_RI)])-0.05
xmax = np.max([np.concatenate(model_RI), np.concatenate(human_RI)])+0.05
plt.plot([xmin, xmax], [xmin, xmax], 'k', linewidth=3)
for i in range(len(folders)):
    for j in range(num_conditions*num_participants):
        current_cond = j%num_conditions
        
        plt.plot(model_RI[i][j], human_RI[i][j], marker=markers[current_cond], color=colors[current_cond]);
plt.xlabel('model RI')
plt.ylabel('human RI')
plt.xlim([xmin, xmax])
plt.ylim([xmin, xmax])
#plt.title('Human vs. model regression indices for all networks')
plt.savefig(os.path.join(evaluation_folder, "human-vs-model-RI.pdf"))
plt.tight_layout()
plt.close()

        
##################################################
# SUMMARY ACROSS GRADUAL CHANGE OF H and sigma^2 #
##################################################

### HYPER-PRIOR case ###

## using H
distance_values = np.empty((len(folders),), dtype=object)
mean_values = np.empty((len(folders),), dtype=object)
fig = plt.figure(figsize = (15,10))
parameters = {'xtick.labelsize': 25,
              'ytick.labelsize': 25,
              'axes.labelsize': 25}
plt.rcParams.update(parameters)
for i,f in enumerate(folders):
    hyper_H_values = np.load('./results/network_summary/hyper_H_values.npy')
    prior_hyper = np.load(os.path.join(f, 'modify-prior_soc-hyper.npy'))
    # first line contains human values: RI_soc_h - RI_mech_h
    # second line contains human values: RI_soc_h - RI_tablet_h
    # third+ lines: model values: social - H-modified
    plt.plot(hyper_H_values, np.tile(np.mean(prior_hyper[2]), np.size(hyper_H_values)), 'k-')
    plt.plot(hyper_H_values, np.tile(np.mean(prior_hyper[0]), np.size(hyper_H_values)), 'k--')
    plt.plot(hyper_H_values, np.tile(np.mean(prior_hyper[1]), np.size(hyper_H_values)), 'k-.')
    distance_values[i] = prior_hyper[2:,:]
    mean_values[i] = np.mean(distance_values[i],axis=1)
    plt.plot(hyper_H_values, mean_values[i], 'x', markersize=20)
plt.xlabel('H parameter')
plt.ylabel('subject-wise difference social(H=1) - social(H=x)')
plt.grid(which='minor')
plt.tight_layout()
plt.savefig(os.path.join(evaluation_folder, 'prior_soc-hyper_means.pdf'))

human_line_1 = np.mean(prior_hyper[0,:])
human_line_2 = np.mean(prior_hyper[1,:])
means_I_H = np.concatenate(mean_values).reshape((mean_values.shape[0],-1)) # => 10 networks x 15 H values
# which is closest to the human_line_1?
print(hyper_H_values[np.argmin(np.mean(np.abs(means_I_H - np.tile(human_line_1, means_I_H.shape)),axis=0))])
print(hyper_H_values[np.argmin(np.mean(np.abs(means_I_H - np.tile(human_line_2, means_I_H.shape)),axis=0))])
best_for_participants_1 = []
best_for_participants_2 = []
# check for each of the networks all the argmin per participant => later take the one that fits most participants
for i, m in enumerate(distance_values):
    # the parameter that best describes the different participants
    best_for_participants_1.append(np.argmin(np.abs(m - np.tile(human_line_1, m.shape)),axis=0))
    best_for_participants_2.append(np.argmin(np.abs(m - np.tile(human_line_2, m.shape)),axis=0))
print("Increasing the precision of the prior (H) to the following values reproduces human data:")
print(hyper_H_values[int(np.mean([np.mean(x) for x in best_for_participants_1]))])
print(hyper_H_values[int(np.mean([np.mean(x) for x in best_for_participants_2]))])

# which network is the closest to these optimal values
# np.argmin(np.abs(means_I_H - np.tile(human_line_1, means_I_H.shape))[:,4] + np.abs(means_I_H - np.tile(human_line_2, means_I_H.shape))[:,9])



# using sigma^2
distance_values = np.empty((len(folders),), dtype=object)
mean_values = np.empty((len(folders),), dtype=object)
fig = plt.figure(figsize = (15,10))
parameters = {'xtick.labelsize': 25,
              'ytick.labelsize': 25,
              'axes.labelsize': 25}
plt.rcParams.update(parameters)
for i,f in enumerate(folders):
    sigma_inc_values = np.load(os.path.join(f, 'increase_sigma_inp.npy'))
    value_sigma_inc = np.load(os.path.join(f, 'modify-sigma-inp_soc-increaseSig.npy'))
    plt.plot(sigma_inc_values, np.tile(np.mean(value_sigma_inc[2]), np.size(sigma_inc_values)), 'k-')
    plt.plot(sigma_inc_values, np.tile(np.mean(value_sigma_inc[0]), np.size(sigma_inc_values)), 'k--')
    plt.plot(sigma_inc_values, np.tile(np.mean(value_sigma_inc[1]), np.size(sigma_inc_values)), 'k-.')
    distance_values[i] = value_sigma_inc[2:,:]
    mean_values[i] = np.mean(distance_values[i],axis=1)
    plt.plot(sigma_inc_values, mean_values[i], 'x', markersize=20)
plt.xlabel('\sigma^2 parameter')
plt.ylabel('subject-wise difference social(\sigma^2=0.001) - social(\sigma^2=x)')
plt.savefig(os.path.join(evaluation_folder, 'prior_sigma-inc_means.pdf'))

human_line_1 = np.mean(value_sigma_inc[0,:])
human_line_2 = np.mean(value_sigma_inc[1,:])
means_I_H = np.concatenate(mean_values).reshape((mean_values.shape[0],-1)) # => 10 networks x 15 H values
# which is closest to the human_line_1?
print(hyper_H_values[np.argmin(np.mean(np.abs(means_I_H - np.tile(human_line_1, means_I_H.shape)),axis=0))])
print(hyper_H_values[np.argmin(np.mean(np.abs(means_I_H - np.tile(human_line_2, means_I_H.shape)),axis=0))])
best_for_participants_1 = []
best_for_participants_2 = []
# check for each of the networks all the argmin per participant => later take the one that fits most participants
for i, m in enumerate(distance_values):
    # the parameter that best describes the different participants
    best_for_participants_1.append(np.argmin(np.abs(m - np.tile(human_line_1, m.shape)),axis=0))
    best_for_participants_2.append(np.argmin(np.abs(m - np.tile(human_line_2, m.shape)),axis=0))
print("Reducing the precision of the input (sigma) by the following values reproduces human data:")
print(sigma_inc_values[int(np.mean([np.mean(x) for x in best_for_participants_1]))])
print(sigma_inc_values[int(np.mean([np.mean(x) for x in best_for_participants_2]))])

### HYPO-PRIOR case ###

distance_values = np.empty((len(folders),), dtype=object)
mean_values = np.empty((len(folders),), dtype=object)
fig = plt.figure(figsize = (15,10))
parameters = {'xtick.labelsize': 25,
              'ytick.labelsize': 25,
              'axes.labelsize': 25}
plt.rcParams.update(parameters)
for i,f in enumerate(folders):
    hypo_H_values = np.load(os.path.join(f, 'hypo_H_values.npy'))
    prior_hypo = np.load(os.path.join(f, 'modify-prior_tablet-hypo.npy'))
    plt.plot(hypo_H_values, np.tile(np.mean(prior_hypo[2]), np.size(hypo_H_values)), 'k-')
    plt.plot(hypo_H_values, np.tile(np.mean(prior_hypo[0]), np.size(hypo_H_values)), 'k--')
    plt.plot(hypo_H_values, np.tile(np.mean(prior_hypo[1]), np.size(hypo_H_values)), 'k-.')
    distance_values[i] = prior_hypo[2:,:]
    mean_values[i] = np.mean(distance_values[i],axis=1)
    plt.plot(hypo_H_values, mean_values[i], 'x', markersize=20)
plt.xlabel('H parameter')
plt.ylabel('subject-wise difference social(H=1) - social(H=x)')
plt.grid(which='minor')
plt.tight_layout()
plt.savefig(os.path.join(evaluation_folder, 'prior_tablet-hypo_means.pdf'))

human_line_1 = np.mean(prior_hypo[0,:])
human_line_2 = np.mean(prior_hypo[1,:])
means_I_H = np.concatenate(mean_values).reshape((mean_values.shape[0],-1)) # => 10 networks x 15 H values
# which is closest to the human_line_1?
print(hypo_H_values[np.argmin(np.mean(np.abs(means_I_H - np.tile(human_line_1, means_I_H.shape)),axis=0))])
print(hypo_H_values[np.argmin(np.mean(np.abs(means_I_H - np.tile(human_line_2, means_I_H.shape)),axis=0))])
best_for_participants_1 = []
best_for_participants_2 = []
# check for each of the networks all the argmin per participant => later take the one that fits most participants
for i, m in enumerate(distance_values):
    # the parameter that best describes the different participants
    best_for_participants_1.append(np.argmin(np.abs(m - np.tile(human_line_1, m.shape)),axis=0))
    best_for_participants_2.append(np.argmin(np.abs(m - np.tile(human_line_2, m.shape)),axis=0))
print("Decreasing the precision of the prior (H) to the following values reproduces human data:")
print(hypo_H_values[int(np.mean([np.mean(x) for x in best_for_participants_1]))])
print(hypo_H_values[int(np.mean([np.mean(x) for x in best_for_participants_2]))])



distance_values = np.empty((len(folders),), dtype=object)
mean_values = np.empty((len(folders),), dtype=object)
fig = plt.figure(figsize = (15,10))
parameters = {'xtick.labelsize': 25,
              'ytick.labelsize': 25,
              'axes.labelsize': 25}
plt.rcParams.update(parameters)
for i,f in enumerate(folders):
    sigma_dec_values = np.load(os.path.join(f, 'decrease_sigma_inp.npy'))
    value_sigma_dec = np.load(os.path.join(f, 'modify-sigma-inp_tablet-decreaseSig.npy'))
    plt.plot(sigma_dec_values, np.tile(np.mean(value_sigma_dec[2]), np.size(sigma_dec_values)), 'k-')
    plt.plot(sigma_dec_values, np.tile(np.mean(value_sigma_dec[0]), np.size(sigma_dec_values)), 'k--')
    plt.plot(sigma_dec_values, np.tile(np.mean(value_sigma_dec[1]), np.size(sigma_dec_values)), 'k-.')
    distance_values[i] = value_sigma_dec[2:,:]
    mean_values[i] = np.mean(distance_values[i],axis=1)
    plt.plot(sigma_dec_values, mean_values[i], 'x', markersize=20)
plt.xlabel('\sigma^2 parameter')
plt.ylabel('subject-wise difference social(\sigma^2=0.001) - social(\sigma^2=x)')
plt.savefig(os.path.join(evaluation_folder, 'prior_sigma-dec_means.pdf'))

human_line_1 = np.mean(value_sigma_dec[0,:])
human_line_2 = np.mean(value_sigma_dec[1,:])
means_I_H = np.concatenate(mean_values).reshape((mean_values.shape[0],-1)) # => 10 networks x 15 H values
# which is closest to the human_line_1?
print(hyper_H_values[np.argmin(np.mean(np.abs(means_I_H - np.tile(human_line_1, means_I_H.shape)),axis=0))])
print(hyper_H_values[np.argmin(np.mean(np.abs(means_I_H - np.tile(human_line_2, means_I_H.shape)),axis=0))])
best_for_participants_1 = []
best_for_participants_2 = []
# check for each of the networks all the argmin per participant => later take the one that fits most participants
for i, m in enumerate(distance_values):
    # the parameter that best describes the different participants
    best_for_participants_1.append(np.argmin(np.abs(m - np.tile(human_line_1, m.shape)),axis=0))
    best_for_participants_2.append(np.argmin(np.abs(m - np.tile(human_line_2, m.shape)),axis=0))
print("Reducing the precision of the input (sigma) by the following values reproduces human data:")
print(sigma_dec_values[int(np.mean([np.mean(x) for x in best_for_participants_1]))])
print(sigma_dec_values[int(np.mean([np.mean(x) for x in best_for_participants_2]))])



############################################
# SUMMARY ACROSS NETWORK ACTIVATION TRACES #
############################################

#aLaP = np.empty((len(folders),),dtype=object)
wLaP = np.empty((len(folders),),dtype=object)
aLwP = np.empty((len(folders),),dtype=object)

for i,f in enumerate(folders):
    #aLaP[i] = np.load(os.path.join(f, 'activations_across-lengths_across-participants_H-' + str(test_H) + '.npy'), allow_pickle=True)
    wLaP[i] = np.load(os.path.join(f, 'activations_within-lengths_across-participants_H-' + str(test_H) + '.npy'), allow_pickle=True)
    aLwP[i] = np.load(os.path.join(f, 'activations_across-lengths_within-participants_H-' + str(test_H) + '.npy'), allow_pickle=True)

# wLaP[num_network][num_array, num_length][num_comparison][num_timestep]
#           10         6           11         e.g.11175         22
# wLaP contains for all len(folders) networks, 6 different arrays (within-0, within-1, within-2, between-0-1, between-0-2, between-1-2)
# 11 length categories
# each with an N x 22 array where 22=time steps, N is the number of trajectory comparisons

# over networks, one array, across length categories
# [i][0]

"""
all_within_0 = np.concatenate(wLaP[0][0])
all_within_1 = np.concatenate(wLaP[0][1])
all_within_2 = np.concatenate(wLaP[0][2])
all_between_0_1 = np.concatenate(wLaP[0][3])
all_between_0_2 = np.concatenate(wLaP[0][4])
all_between_1_2 = np.concatenate(wLaP[0][5])
for i in range(1, len(folders)):
    all_within_0 = np.concatenate((all_within_0, np.concatenate(wLaP[i][0])))
    all_within_1 = np.concatenate((all_within_0, np.concatenate(wLaP[i][1])))
    all_within_2 = np.concatenate((all_within_0, np.concatenate(wLaP[i][2])))
    all_between_0_1 = np.concatenate((all_within_0, np.concatenate(wLaP[i][3])))
    all_between_0_2 = np.concatenate((all_within_0, np.concatenate(wLaP[i][4])))
    all_between_1_2 = np.concatenate((all_within_0, np.concatenate(wLaP[i][5])))
"""

all_within_0 = np.mean(np.concatenate(wLaP[0][0]),axis=0).reshape((1,-1))
all_within_1 = np.mean(np.concatenate(wLaP[0][1]),axis=0).reshape((1,-1))
all_within_2 = np.mean(np.concatenate(wLaP[0][2]),axis=0).reshape((1,-1))
all_between_0_1 = np.mean(np.concatenate(wLaP[0][3]),axis=0).reshape((1,-1))
all_between_0_2 = np.mean(np.concatenate(wLaP[0][4]),axis=0).reshape((1,-1))
all_between_1_2 = np.mean(np.concatenate(wLaP[0][5]),axis=0).reshape((1,-1))
for i in range(1, len(folders)):
    all_within_0 = np.concatenate((all_within_0, np.mean(np.concatenate(wLaP[i][0]),axis=0).reshape((1,-1))),axis=0)
    all_within_1 = np.concatenate((all_within_1, np.mean(np.concatenate(wLaP[i][1]),axis=0).reshape((1,-1))),axis=0)
    all_within_2 = np.concatenate((all_within_2, np.mean(np.concatenate(wLaP[i][2]),axis=0).reshape((1,-1))),axis=0)
    all_between_0_1 = np.concatenate((all_between_0_1, np.mean(np.concatenate(wLaP[i][3]),axis=0).reshape((1,-1))),axis=0)
    all_between_0_2 = np.concatenate((all_between_0_2, np.mean(np.concatenate(wLaP[i][4]),axis=0).reshape((1,-1))),axis=0)
    all_between_1_2 = np.concatenate((all_between_1_2, np.mean(np.concatenate(wLaP[i][5]),axis=0).reshape((1,-1))),axis=0)

plt.figure()
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_0,axis=0), yerr=(np.std(all_within_0,axis=0) / np.sqrt(all_within_0.shape[0])), color='r', label='tablet')
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_1,axis=0), yerr=(np.std(all_within_1,axis=0) / np.sqrt(all_within_1.shape[0])), color='y', label='mechanical')
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_2,axis=0), yerr=(np.std(all_within_2,axis=0) / np.sqrt(all_within_2.shape[0])), color='b', label='social')

#plt.errorbar(np.arange(num_timesteps), np.mean(all_between_0_1,axis=0), yerr=(np.std(all_between_0_1,axis=0) / np.sqrt(all_between_0_1.shape[0])), color='orange', linestyle='--', label='tablet-mechanical')
#plt.errorbar(np.arange(num_timesteps), np.mean(all_between_0_2,axis=0), yerr=(np.std(all_between_0_2,axis=0) / np.sqrt(all_between_0_2.shape[0])), color='purple', linestyle='--', label='tablet-social')
#plt.errorbar(np.arange(num_timesteps), np.mean(all_between_1_2,axis=0), yerr=(np.std(all_between_1_2,axis=0) / np.sqrt(all_between_1_2.shape[0])), color='g', linestyle='--', label='mechanical-social')
plt.legend()
plt.xlabel('time steps')
plt.ylabel('average distance')
plt.savefig(os.path.join(evaluation_folder, 'Summary_across_networks_wLaP_' + str(train_H) + "_" + str(test_H) + '.png'))
plt.close()


# write text file for plotting
Tmean = np.mean(all_within_0,axis=0)
Tse = (np.std(all_within_0,axis=0) / np.sqrt(all_within_0.shape[0]))
Mmean = np.mean(all_within_1,axis=0)
Mse = (np.std(all_within_1,axis=0) / np.sqrt(all_within_1.shape[0]))
Smean = np.mean(all_within_2,axis=0)
Sse = (np.std(all_within_2,axis=0) / np.sqrt(all_within_2.shape[0]))

with open(os.path.join(evaluation_folder, "wLaP_results.txt"), 'w') as f:
    f.write("t\tTmean\tTse\tMmean\tMse\tSmean\tSse\n")
    for t in range(num_timesteps):
        f.write(str(t) + "\t" + str(Tmean[t]) + "\t" + str(Tse[t]) + "\t" + str(Mmean[t]) + "\t" + str(Mse[t]) + "\t" + str(Smean[t]) + "\t" + str(Sse[t]) + "\n")
    
# write text file for doing statistical analysis
with open(os.path.join(evaluation_folder, "wLaP_statistics.csv"), 'w') as f:
    f.write("net\tcondition\ttimestep\tdistance\n")
    for net in range(all_within_0.shape[0]):
        for t in range(num_timesteps):
            f.write(str(net) + "\t0\t" + str(t) + "\t" + str(all_within_0[net,t]) + "\n")
            f.write(str(net) + "\t1\t" + str(t) + "\t" + str(all_within_1[net,t]) + "\n")
            f.write(str(net) + "\t2\t" + str(t) + "\t" + str(all_within_2[net,t]) + "\n")


"""
# perform the analysis normalized per network
values_new_0 = np.empty((len(activations),), dtype=object)
values_new_1 = np.empty((len(activations),), dtype=object)
values_new_2 = np.empty((len(activations),), dtype=object)
min_target = 0
max_target = 1
for i in range(len(activations)):
    values = np.concatenate(activations[i][0,:], axis=0)
    min_val = np.min(values)
    max_val = np.max(values)
    values_new_0[i] = (values - min_val) * (max_target - min_target) / (max_val - min_val) + min_target

    values = np.concatenate(activations[i][1,:], axis=0)
    min_val = np.min(values)
    max_val = np.max(values)
    values_new_1[i] = (values - min_val) * (max_target - min_target) / (max_val - min_val) + min_target

    values = np.concatenate(activations[i][2,:], axis=0)
    min_val = np.min(values)
    max_val = np.max(values)
    values_new_2[i] = (values - min_val) * (max_target - min_target) / (max_val - min_val) + min_target

# 0: tablet, 1: mechanical, 2: social condition
all_within_0 = np.mean(values_new_0[0],axis=0).reshape((1,-1))
all_within_1 = np.mean(values_new_1[0],axis=0).reshape((1,-1))
all_within_2 = np.mean(values_new_2[0],axis=0).reshape((1,-1))
for i in range(1, len(activations)):
    all_within_0 = np.concatenate((all_within_0, np.mean(values_new_0[i],axis=0).reshape((1,-1))),axis=0)
    all_within_1 = np.concatenate((all_within_1, np.mean(values_new_1[i],axis=0).reshape((1,-1))),axis=0)
    all_within_2 = np.concatenate((all_within_2, np.mean(values_new_2[i],axis=0).reshape((1,-1))),axis=0)

plt.figure()
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_0,axis=0), yerr=(np.std(all_within_0,axis=0) / np.sqrt(all_within_0.shape[0])), color='r', label='tablet')
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_1,axis=0), yerr=(np.std(all_within_1,axis=0) / np.sqrt(all_within_1.shape[0])), color='y', label='mechanical')
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_2,axis=0), yerr=(np.std(all_within_2,axis=0) / np.sqrt(all_within_2.shape[0])), color='b', label='social')

#plt.errorbar(np.arange(num_timesteps), np.mean(all_between_0_1,axis=0), yerr=(np.std(all_between_0_1,axis=0) / np.sqrt(all_between_0_1.shape[0])), color='orange', linestyle='--', label='tablet-mechanical')
#plt.errorbar(np.arange(num_timesteps), np.mean(all_between_0_2,axis=0), yerr=(np.std(all_between_0_2,axis=0) / np.sqrt(all_between_0_2.shape[0])), color='purple', linestyle='--', label='tablet-social')
#plt.errorbar(np.arange(num_timesteps), np.mean(all_between_1_2,axis=0), yerr=(np.std(all_between_1_2,axis=0) / np.sqrt(all_between_1_2.shape[0])), color='g', linestyle='--', label='mechanical-social')
plt.legend()
plt.xlabel('time steps')
plt.ylabel('average distance')
plt.savefig(os.path.join(evaluation_folder, 'Summary_across_networks_wLaP_' + str(train_H) + "_" + str(test_H) + '_norm.png'))
plt.close()

# write text file for plotting
Tmean = np.mean(all_within_0,axis=0)
Tse = (np.std(all_within_0,axis=0) / np.sqrt(all_within_0.shape[0]))
Mmean = np.mean(all_within_1,axis=0)
Mse = (np.std(all_within_1,axis=0) / np.sqrt(all_within_1.shape[0]))
Smean = np.mean(all_within_2,axis=0)
Sse = (np.std(all_within_2,axis=0) / np.sqrt(all_within_2.shape[0]))

with open(os.path.join(evaluation_folder, "wLaP_results_norm.txt"), 'w') as f:
    f.write("t\tTmean\tTse\tMmean\tMse\tSmean\tSse\n")
    for t in range(num_timesteps):
        f.write(str(t) + "\t" + str(Tmean[t]) + "\t" + str(Tse[t]) + "\t" + str(Mmean[t]) + "\t" + str(Mse[t]) + "\t" + str(Smean[t]) + "\t" + str(Sse[t]) + "\n")
    
# write text file for doing statistical analysis
with open(os.path.join(evaluation_folder, "wLaP_statistics_norm.csv"), 'w') as f:
    f.write("net\tcondition\ttimestep\tdistance\n")
    for net in range(all_within_0.shape[0]):
        for t in range(num_timesteps):
            f.write(str(net) + "\t0\t" + str(t) + "\t" + str(all_within_0[net,t]) + "\n")
            f.write(str(net) + "\t1\t" + str(t) + "\t" + str(all_within_1[net,t]) + "\n")
            f.write(str(net) + "\t2\t" + str(t) + "\t" + str(all_within_2[net,t]) + "\n")
"""



#np.concatenate(np.concatenate(wLaP[0][0:3,:],axis=0))

# all_within_0 => 10x22: the 10 mean values for the 10 networks, per time step

# tablet is supposed to be higher than mechanical, and higher than social in the first step
# social is supposed to 
between_dists_0_1 = (all_within_0 - all_within_1)
between_dists_0_2 = (all_within_0 - all_within_2)
between_dists_1_2 = (all_within_1 - all_within_2)

# write text file for doing statistical analysis
with open(os.path.join(evaluation_folder, "wLaP-between_statistics.csv"), 'w') as f:
    f.write("condition\ttimestep\tdistance\n")
    for net in range(all_within_0.shape[0]):
        for t in range(num_timesteps):
            f.write("0\t" + str(t) + "\t" + str(between_dists_0_1[net,t]) + "\n")
            f.write("1\t" + str(t) + "\t" + str(between_dists_0_2[net,t]) + "\n")
            f.write("2\t" + str(t) + "\t" + str(between_dists_1_2[net,t]) + "\n")


plt.figure()
plt.errorbar(np.arange(num_timesteps), np.mean(between_dists_0_1,axis=0), yerr=(np.std(between_dists_0_1,axis=0) / np.sqrt(between_dists_0_1.shape[0])), color='orange', linestyle='--', label='tablet-mechanical')
plt.errorbar(np.arange(num_timesteps), np.mean(between_dists_0_2,axis=0), yerr=(np.std(between_dists_0_2,axis=0) / np.sqrt(between_dists_0_2.shape[0])), color='purple', linestyle='--', label='tablet-social')
plt.errorbar(np.arange(num_timesteps), np.mean(between_dists_1_2,axis=0), yerr=(np.std(between_dists_1_2,axis=0) / np.sqrt(between_dists_1_2.shape[0])), color='g', linestyle='--', label='mechanical-social')
plt.legend()
plt.xlabel('time steps')
plt.ylabel('average distance')
plt.savefig(os.path.join(evaluation_folder, 'Summary_across_networks_wLaP_dist-between-networks_' + str(train_H) + "_" + str(test_H) + '.png'))
plt.close()

# write text file for plotting
Tmean = np.mean(between_dists_0_1,axis=0)
Tse = (np.std(between_dists_0_1,axis=0) / np.sqrt(between_dists_0_1.shape[0]))
Mmean = np.mean(between_dists_0_2,axis=0)
Mse = (np.std(between_dists_0_2,axis=0) / np.sqrt(between_dists_0_2.shape[0]))
Smean = np.mean(between_dists_1_2,axis=0)
Sse = (np.std(between_dists_1_2,axis=0) / np.sqrt(between_dists_1_2.shape[0]))

with open(os.path.join(evaluation_folder, "wLaP-between_results.txt"), 'w') as f:
    f.write("t\tTmean\tTse\tMmean\tMse\tSmean\tSse\n")
    for t in range(num_timesteps):
        f.write(str(t) + "\t" + str(Tmean[t]) + "\t" + str(Tse[t]) + "\t" + str(Mmean[t]) + "\t" + str(Mse[t]) + "\t" + str(Smean[t]) + "\t" + str(Sse[t]) + "\n")
    
# write text file for doing statistical analysis
with open(os.path.join(evaluation_folder, "wLaP-between_statistics.csv"), 'w') as f:
    f.write("net\tcondition\ttimestep\tdistance\n")
    for net in range(between_dists_0_1.shape[0]):
        for t in range(num_timesteps):
            f.write(str(net) + "\t0\t" + str(t) + "\t" + str(between_dists_0_1[net,t]) + "\n")
            f.write(str(net) + "\t1\t" + str(t) + "\t" + str(between_dists_0_2[net,t]) + "\n")
            f.write(str(net) + "\t2\t" + str(t) + "\t" + str(between_dists_1_2[net,t]) + "\n")











"""
all_within_0 = np.concatenate(aLwP[0][0])
all_within_1 = np.concatenate(aLwP[0][1])
all_within_2 = np.concatenate(aLwP[0][2])
all_between_0_1 = np.concatenate(aLwP[0][3])
all_between_0_2 = np.concatenate(aLwP[0][4])
all_between_1_2 = np.concatenate(aLwP[0][5])
for i in range(1, len(folders)):
    all_within_0 = np.concatenate((all_within_0, np.concatenate(aLwP[i][0])))
    all_within_1 = np.concatenate((all_within_0, np.concatenate(aLwP[i][1])))
    all_within_2 = np.concatenate((all_within_0, np.concatenate(aLwP[i][2])))
    all_between_0_1 = np.concatenate((all_within_0, np.concatenate(aLwP[i][3])))
    all_between_0_2 = np.concatenate((all_within_0, np.concatenate(aLwP[i][4])))
    all_between_1_2 = np.concatenate((all_within_0, np.concatenate(aLwP[i][5])))
"""

all_within_0 = np.mean(np.concatenate(aLwP[0][0]),axis=0).reshape((1,-1))
all_within_1 = np.mean(np.concatenate(aLwP[0][1]),axis=0).reshape((1,-1))
all_within_2 = np.mean(np.concatenate(aLwP[0][2]),axis=0).reshape((1,-1))
all_between_0_1 = np.mean(np.concatenate(aLwP[0][3]),axis=0).reshape((1,-1))
all_between_0_2 = np.mean(np.concatenate(aLwP[0][4]),axis=0).reshape((1,-1))
all_between_1_2 = np.mean(np.concatenate(aLwP[0][5]),axis=0).reshape((1,-1))
for i in range(1, len(folders)):
    all_within_0 = np.concatenate((all_within_0, np.mean(np.concatenate(aLwP[i][0]),axis=0).reshape((1,-1))),axis=0)
    all_within_1 = np.concatenate((all_within_1, np.mean(np.concatenate(aLwP[i][1]),axis=0).reshape((1,-1))),axis=0)
    all_within_2 = np.concatenate((all_within_2, np.mean(np.concatenate(aLwP[i][2]),axis=0).reshape((1,-1))),axis=0)
    all_between_0_1 = np.concatenate((all_between_0_1, np.mean(np.concatenate(aLwP[i][3]),axis=0).reshape((1,-1))),axis=0)
    all_between_0_2 = np.concatenate((all_between_0_2, np.mean(np.concatenate(aLwP[i][4]),axis=0).reshape((1,-1))),axis=0)
    all_between_1_2 = np.concatenate((all_between_1_2, np.mean(np.concatenate(aLwP[i][5]),axis=0).reshape((1,-1))),axis=0)

plt.figure()
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_0,axis=0), yerr=(np.std(all_within_0,axis=0) / np.sqrt(all_within_0.shape[0])), color='r', label='tablet')
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_1,axis=0), yerr=(np.std(all_within_1,axis=0) / np.sqrt(all_within_1.shape[0])), color='y', label='mechanical')
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_2,axis=0), yerr=(np.std(all_within_2,axis=0) / np.sqrt(all_within_2.shape[0])), color='b', label='social')

plt.errorbar(np.arange(num_timesteps), np.mean(all_between_0_1,axis=0), yerr=(np.std(all_between_0_1,axis=0) / np.sqrt(all_between_0_1.shape[0])), color='orange', linestyle='--', label='tablet-mechanical')
plt.errorbar(np.arange(num_timesteps), np.mean(all_between_0_2,axis=0), yerr=(np.std(all_between_0_2,axis=0) / np.sqrt(all_between_0_2.shape[0])), color='purple', linestyle='--', label='tablet-social')
plt.errorbar(np.arange(num_timesteps), np.mean(all_between_1_2,axis=0), yerr=(np.std(all_between_1_2,axis=0) / np.sqrt(all_between_1_2.shape[0])), color='g', linestyle='--', label='mechanical-social')
plt.legend()
plt.xlabel('time steps')
plt.ylabel('average distance')
plt.savefig(os.path.join(evaluation_folder, 'Summary_across_networks_aLwP_' + str(train_H) + "_" + str(test_H) + '.png'))
plt.close()








"""

all_within_0 = np.mean(np.concatenate(aLaP[0][0]),axis=0).reshape((1,-1))
all_within_1 = np.mean(np.concatenate(aLaP[0][1]),axis=0).reshape((1,-1))
all_within_2 = np.mean(np.concatenate(aLaP[0][2]),axis=0).reshape((1,-1))
all_between_0_1 = np.mean(np.concatenate(aLaP[0][3]),axis=0).reshape((1,-1))
all_between_0_2 = np.mean(np.concatenate(aLaP[0][4]),axis=0).reshape((1,-1))
all_between_1_2 = np.mean(np.concatenate(aLaP[0][5]),axis=0).reshape((1,-1))
for i in range(1, len(folders)):
    all_within_0 = np.concatenate((all_within_0, np.mean(np.concatenate(aLaP[i][0]),axis=0).reshape((1,-1))),axis=0)
    all_within_1 = np.concatenate((all_within_1, np.mean(np.concatenate(aLaP[i][1]),axis=0).reshape((1,-1))),axis=0)
    all_within_2 = np.concatenate((all_within_2, np.mean(np.concatenate(aLaP[i][2]),axis=0).reshape((1,-1))),axis=0)
    all_between_0_1 = np.concatenate((all_between_0_1, np.mean(np.concatenate(aLaP[i][3]),axis=0).reshape((1,-1))),axis=0)
    all_between_0_2 = np.concatenate((all_between_0_2, np.mean(np.concatenate(aLaP[i][4]),axis=0).reshape((1,-1))),axis=0)
    all_between_1_2 = np.concatenate((all_between_1_2, np.mean(np.concatenate(aLaP[i][5]),axis=0).reshape((1,-1))),axis=0)

plt.figure()
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_0,axis=0), yerr=(np.std(all_within_0,axis=0) / np.sqrt(all_within_0.shape[0])), color='r', label='tablet')
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_1,axis=0), yerr=(np.std(all_within_1,axis=0) / np.sqrt(all_within_1.shape[0])), color='y', label='mechanical')
plt.errorbar(np.arange(num_timesteps), np.mean(all_within_2,axis=0), yerr=(np.std(all_within_2,axis=0) / np.sqrt(all_within_2.shape[0])), color='b', label='social')

plt.errorbar(np.arange(num_timesteps), np.mean(all_between_0_1,axis=0), yerr=(np.std(all_between_0_1,axis=0) / np.sqrt(all_between_0_1.shape[0])), color='orange', linestyle='--', label='tablet-mechanical')
plt.errorbar(np.arange(num_timesteps), np.mean(all_between_0_2,axis=0), yerr=(np.std(all_between_0_2,axis=0) / np.sqrt(all_between_0_2.shape[0])), color='purple', linestyle='--', label='tablet-social')
plt.errorbar(np.arange(num_timesteps), np.mean(all_between_1_2,axis=0), yerr=(np.std(all_between_1_2,axis=0) / np.sqrt(all_between_1_2.shape[0])), color='g', linestyle='--', label='mechanical-social')
plt.legend()
plt.xlabel('time steps')
plt.ylabel('average distance')
plt.savefig(os.path.join(evaluation_folder, 'Summary_across_networks_aLaP_' + str(train_H) + "_" + str(test_H) + '.png'))
plt.close()
"""




