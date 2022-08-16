import numpy as np
import matplotlib.pyplot as plt

num_timesteps=22

activations = np.load('results/network-activation-summary.npy')

"""
activations.shape => (10,)
These are the ten networks across which we want to average.

activations[0].shape => (6, 11)
6 different measures were computed, the distances...
  0: within the activations of the tablet conditions
  1: within the activations of the mechanical condition
  2: within the activations of the social condition
  3: between tablet and mechanical condition activations
  4: between tablet and social condition activations
  5: between mechanical and social condition activations.
We are mainly interested in the first three measures which correspond to the three lines (red, yellow and blue) in Fig.9 of the paper.
11 is the number of length categories. Distances were only computed within trajectories of the same length category, therefore, they are separately listed. But we don't need the separation anymore, so we can simply concatenate them here.

np.concatenate(activations[0][0,:], axis=0) gives us all the within-tablet-activations distances.

np.concatenate(activations[0][0,:], axis=0).shape => (122628, 22)
The first dimension is the number of comparisons that were made which is very high as every drawn trajectory in this condition (about 1500 because there are 25 participants and they produce about 66 trajectories each) is compared to any other drawn trajectory in this condition.
22 is the number of time steps.

We currently average across all the ca. 122628 values per network.

The plot shows the mean and standard error between the networks as shown in Fig.10 of the draft.
"""

# 0: tablet, 1: mechanical, 2: social condition
all_within_0 = np.mean(np.concatenate(activations[0][0,:]),axis=0).reshape((1,-1))
all_within_1 = np.mean(np.concatenate(activations[0][1,:]),axis=0).reshape((1,-1))
all_within_2 = np.mean(np.concatenate(activations[0][2,:]),axis=0).reshape((1,-1))
all_between_0_1 = np.mean(np.concatenate(activations[0][3,:]),axis=0).reshape((1,-1))
all_between_0_2 = np.mean(np.concatenate(activations[0][4,:]),axis=0).reshape((1,-1))
all_between_1_2 = np.mean(np.concatenate(activations[0][5,:]),axis=0).reshape((1,-1))
for i in range(1, len(activations)):
    all_within_0 = np.concatenate((all_within_0, np.mean(np.concatenate(activations[i][0,:]),axis=0).reshape((1,-1))),axis=0)
    all_within_1 = np.concatenate((all_within_1, np.mean(np.concatenate(activations[i][1,:]),axis=0).reshape((1,-1))),axis=0)
    all_within_2 = np.concatenate((all_within_2, np.mean(np.concatenate(activations[i][2,:]),axis=0).reshape((1,-1))),axis=0)
    all_between_0_1 = np.concatenate((all_between_0_1, np.mean(np.concatenate(activations[i][3,:]),axis=0).reshape((1,-1))),axis=0)
    all_between_0_2 = np.concatenate((all_between_0_2, np.mean(np.concatenate(activations[i][4,:]),axis=0).reshape((1,-1))),axis=0)
    all_between_1_2 = np.concatenate((all_between_1_2, np.mean(np.concatenate(activations[i][5,:]),axis=0).reshape((1,-1))),axis=0)

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
plt.savefig('Summary_across_networks.png')
plt.close()

# can we normalize the values per network?
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
plt.savefig('Summary_across_networks_norm.png')
plt.close()




