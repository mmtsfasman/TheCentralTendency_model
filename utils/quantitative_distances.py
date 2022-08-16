import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def identify_lengths(presented_lengths, list_of_items_to_cut):
    unique_lengths = np.unique(presented_lengths)
    # use the most common lengths for categorization
    counter_unique_lengths = [np.sum(presented_lengths==u) for u in unique_lengths]
    main_lengths = unique_lengths[np.asarray(counter_unique_lengths)>100]

    # main_lengths_idx = np.where(np.asarray(counter_unique_lengths)>100)[0]
    # histogram = [np.sum(presented_lengths==u) for u in unique_lengths]

    # threshold for the presented length – when to still count it to the main length?
    # average distance between two main lengths
    threshold = np.mean([main_lengths[i]-main_lengths[i-1] for i in range(1,11)])/2

    idc_trials_with_valid_lengths = np.where(np.asarray([np.min(np.sqrt((p - main_lengths)**2))  for p in presented_lengths])<threshold)

    # get the valid lengths, mapped to the closest length
    valid_trials = np.where(np.asarray([np.min(np.sqrt((p - main_lengths)**2)) for p in presented_lengths])<threshold)[0]
    presented_lengths_valid = presented_lengths[valid_trials]
    # get the main lengths corresponding to those presented lengths
    presented_lengths_valid_main = [main_lengths[np.argmin(np.sqrt((p - main_lengths)**2))] for p in presented_lengths_valid]


    cut_items = []
    for item in list_of_items_to_cut:
        if item.ndim == 1:
            cut_items.append(item[valid_trials])
        else:
            cut_items.append(item[valid_trials,:])

    return valid_trials, presented_lengths_valid_main, main_lengths, cut_items


def sort_by_length_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22, scaler=None, pca=None):

    num_lengths = len(main_lengths)

    # 11 x (N x (70*22)), N = number of length reproductions in this bin
    data_cond0 = np.empty((num_lengths,),dtype=object)
    data_cond1 = np.empty((num_lengths,),dtype=object)
    data_cond2 = np.empty((num_lengths,),dtype=object)

    for i in range(num_lengths):
        data_cond0[i] = []
        data_cond1[i] = []
        data_cond2[i] = []

    # separate data per condition
    for i,is_id in enumerate(classes_train_split):
        current_cond = cond_list_split[i] # is_id%num_conditions
        current_subj = subj_list_split[i] # int((is_id - is_id%num_conditions)/num_conditions)
        #print("{} (condition: {}, subject: {})".format(i, current_cond, current_subj))
        length_bin = np.argmin(np.sqrt((presented_lengths_valid_main[i] - main_lengths)**2))

        if pca and scaler:
            current_sample = np.reshape(pca.transform(scaler.transform(u_h_history_split[i,:].reshape((num_timesteps,-1)))), (-1,))
        else:

            current_sample = u_h_history_split[i,:]

        if current_cond == 0:
            data_cond0[length_bin].append(current_sample)
        elif current_cond == 1:
            data_cond1[length_bin].append(current_sample)
        elif current_cond == 2:
            data_cond2[length_bin].append(current_sample)

    for i in range(num_lengths):
        data_cond0[i] = np.asarray(data_cond0[i])
        data_cond1[i] = np.asarray(data_cond1[i])
        data_cond2[i] = np.asarray(data_cond2[i])

    return data_cond0, data_cond1, data_cond2



def sort_by_participant_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22, scaler=None, pca=None):

    num_participants = np.max(subj_list_split)+1

    # 11 x (N x (70*22)), N = number of length reproductions in this bin
    data_cond0 = np.empty((num_participants,),dtype=object)
    data_cond1 = np.empty((num_participants,),dtype=object)
    data_cond2 = np.empty((num_participants,),dtype=object)

    for i in range(num_participants):
        data_cond0[i] = []
        data_cond1[i] = []
        data_cond2[i] = []

    # separate data per condition
    for i,is_id in enumerate(classes_train_split):
        current_cond = cond_list_split[i] # is_id%num_conditions
        current_subj = subj_list_split[i] # int((is_id - is_id%num_conditions)/num_conditions)

        if pca and scaler:
            current_sample = np.reshape(pca.transform(scaler.transform(u_h_history_split[i,:].reshape((num_timesteps,-1)))), (-1,))
        else:

            current_sample = u_h_history_split[i,:]

        if current_cond == 0:
            data_cond0[current_subj].append(current_sample)
        elif current_cond == 1:
            data_cond1[current_subj].append(current_sample)
        elif current_cond == 2:
            data_cond2[current_subj].append(current_sample)

    for i in range(num_participants):
        data_cond0[i] = np.asarray(data_cond0[i])
        data_cond1[i] = np.asarray(data_cond1[i])
        data_cond2[i] = np.asarray(data_cond2[i])

    return data_cond0, data_cond1, data_cond2


def evaluate_activations(u_h_history, init_states, cond_list, subj_list, classes_train, evaluation_folder, filename, num_timesteps = 22, num_participants = 25, num_conditions = 3):
    """
    Do PCA of the context activations, plot the initial state, first and last time step.
    Returns: PCA object, scaler object to be used to project new data into the PCA space.
    """
    # PCA of internal representations: num_samples x (num_timesteps * num_neurons)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # collect all data in all time steps
    all_activations = u_h_history.reshape((u_h_history.shape[0]*num_timesteps,-1))
    all_activations = np.concatenate((all_activations, init_states),axis=0)
    # create the scaler and scale data
    scaler = StandardScaler()
    scaler.fit(all_activations)
    all_act_normed = scaler.transform(all_activations)
    # PCA fit to data
    pca = PCA(n_components=2)
    pca = pca.fit(all_act_normed)
    #all_act_pca = pca.transform(all_act_normed)

    # PLOT: all participants separately #
    #####################################

    fig = plt.figure(figsize = (25,25))
    axObj = np.empty((num_participants,),dtype=object)
    colors = ['r', 'y', 'b']

    first_step_pca = np.zeros((len(classes_train),2))
    last_step_pca = np.zeros((len(classes_train),2))
    for i,is_id in enumerate(classes_train):
        current_cond = cond_list[i] # is_id%num_conditions
        current_subj = subj_list[i] # int((is_id - is_id%num_conditions)/num_conditions)
        pca_per_generated_sample = pca.transform(scaler.transform(u_h_history[i,:].reshape((num_timesteps,-1))))

        # store first step for later comparison to ideal initial states
        first_step_pca[i,:] = pca_per_generated_sample[0,:]
        last_step_pca[i,:] = pca_per_generated_sample[-1,:]

        # for knowing to which participant it belongs (=> select subplot)
        #current_subj = int((is_id - is_id%num_conditions)/num_conditions)
        if not axObj[current_subj]:
            axObj[current_subj] = fig.add_subplot(5,5,current_subj+1)

        # for knowing to which condition it belongs (=> coloring)
        #current_cond = is_id%num_conditions

        axObj[current_subj].scatter(pca_per_generated_sample[:,0], pca_per_generated_sample[:,1], marker='*', color=colors[current_cond])

        # mark the first time step
        axObj[current_subj].scatter(pca_per_generated_sample[0,0], pca_per_generated_sample[0,1], marker='s', color=colors[current_cond])

    plt.savefig(os.path.join(evaluation_folder, "Pca-per-participant-results_" + filename + ".png"))
    plt.close()


    # PLOT: all conditions separately #
    ###################################

    fig = plt.figure(figsize = (30,10))
    axObj = np.empty((num_conditions,),dtype=object)
    colors = sns.color_palette('husl', n_colors=num_participants)

    first_step_pca = np.zeros((len(classes_train),2))
    last_step_pca = np.zeros((len(classes_train),2))
    for i,is_id in enumerate(classes_train):
        current_cond = cond_list[i] # is_id%num_conditions
        current_subj = subj_list[i] # int((is_id - is_id%num_conditions)/num_conditions)
        pca_per_generated_sample = pca.transform(scaler.transform(u_h_history[i,:].reshape((num_timesteps,-1))))

        # store first step for later comparison to ideal initial states
        first_step_pca[i,:] = pca_per_generated_sample[0,:]
        last_step_pca[i,:] = pca_per_generated_sample[-1,:]

        # for knowing to which participant it belongs (=> select subplot)
        #current_subj = int((is_id - is_id%num_conditions)/num_conditions)
        # for knowing to which condition it belongs (=> coloring)
        #current_cond = is_id%num_conditions
        if not axObj[current_cond]:
            axObj[current_cond] = fig.add_subplot(1,3,current_cond+1)
        axObj[current_cond].scatter(pca_per_generated_sample[:,0], pca_per_generated_sample[:,1], marker='*', color=colors[current_subj])

        # mark the first time step
        axObj[current_cond].scatter(pca_per_generated_sample[0,0], pca_per_generated_sample[0,1], marker='s', color=colors[current_subj])

    plt.savefig(os.path.join(evaluation_folder, "Pca-per-condition-results_" + filename + ".png"))
    plt.close()

    return pca, scaler


def evaluate_activations_distances(data_cond0, data_cond1, data_cond2, evaluation_folder, filename, num_neurons=25, num_timesteps=22, plot_indiv = False, ymax = None):
    """
    Given sets of time varying data for the three conditions, computes the distances within and between the conditions,
    and plots these value over the course of trajectory.
    The data structures of the condition data determine whether distances are averaged are computed within all data of
    one condition, or separated in groups. E.g. if data_cond0 is sorted into different arrays by participants, only
    distances within the same participant are considered, and no distances between participants are computed.
    plot_indiv switches on/off the plots of the individual particants or lengths, depending on the separation of the data structure data_cond0 etc.
    Returns the computed distances in the following order: uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2.
    """

    if not filename.endswith('.png'):
        filename += '.png'

    num_lengths = len(data_cond0)
    uh_dists_0_inner = np.empty((num_lengths,), dtype=object)
    uh_dists_1_inner = np.empty((num_lengths,), dtype=object)
    uh_dists_2_inner = np.empty((num_lengths,), dtype=object)
    uh_dists_0_to_1 = np.empty((num_lengths,), dtype=object)
    uh_dists_0_to_2 = np.empty((num_lengths,), dtype=object)
    uh_dists_1_to_2 = np.empty((num_lengths,), dtype=object)
    for l in range(num_lengths):
        #pdist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data_cond0[l]))

        # condition 0 to itself #
        #########################
        uh_dists_0_inner[l] = []
        for i in range(data_cond0[l].shape[0]):
            for j in range(data_cond0[l].shape[0]):
                if i < j:
                    traj1 = data_cond0[l][i,:].reshape((num_timesteps,-1))
                    traj2 = data_cond0[l][j,:].reshape((num_timesteps,-1))
                    dist_t = 1/num_neurons * np.sum(np.sqrt(((traj2-traj1)**2)), axis=1)
                    uh_dists_0_inner[l].append(dist_t)
        uh_dists_0_inner[l] = np.asarray(uh_dists_0_inner[l])

        # condition 1 to itself #
        #########################
        uh_dists_1_inner[l] = []
        for i in range(data_cond1[l].shape[0]):
            for j in range(data_cond1[l].shape[0]):
                if i < j:
                    traj1 = data_cond1[l][i,:].reshape((num_timesteps,-1))
                    traj2 = data_cond1[l][j,:].reshape((num_timesteps,-1))
                    dist_t = 1/num_neurons * np.sum(np.sqrt(((traj2-traj1)**2)), axis=1)
                    uh_dists_1_inner[l].append(dist_t)
        uh_dists_1_inner[l] = np.asarray(uh_dists_1_inner[l])

        # condition 2 to itself #
        #########################
        uh_dists_2_inner[l] = []
        for i in range(data_cond2[l].shape[0]):
            for j in range(data_cond2[l].shape[0]):
                if i < j:
                    traj1 = data_cond2[l][i,:].reshape((num_timesteps,-1))
                    traj2 = data_cond2[l][j,:].reshape((num_timesteps,-1))
                    dist_t = 1/num_neurons * np.sum(np.sqrt(((traj2-traj1)**2)), axis=1)
                    uh_dists_2_inner[l].append(dist_t)
        uh_dists_2_inner[l] = np.asarray(uh_dists_2_inner[l])

        # between conditions #
        ######################
        # condition 0 to 1
        uh_dists_0_to_1[l] = []
        for i in range(data_cond0[l].shape[0]):
            for j in range(data_cond1[l].shape[0]):
                traj1 = data_cond0[l][i,:].reshape((num_timesteps,-1))
                traj2 = data_cond1[l][j,:].reshape((num_timesteps,-1))
                dist_t = 1/num_neurons * np.sum(np.sqrt(((traj2-traj1)**2)), axis=1)
                uh_dists_0_to_1[l].append(dist_t)
        uh_dists_0_to_1[l] = np.asarray(uh_dists_0_to_1[l])

        # condition 0 to 2
        uh_dists_0_to_2[l] = []
        for i in range(data_cond0[l].shape[0]):
            for j in range(data_cond2[l].shape[0]):
                traj1 = data_cond0[l][i,:].reshape((num_timesteps,-1))
                traj2 = data_cond2[l][j,:].reshape((num_timesteps,-1))
                dist_t = 1/num_neurons * np.sum(np.sqrt(((traj2-traj1)**2)), axis=1)
                uh_dists_0_to_2[l].append(dist_t)
        uh_dists_0_to_2[l] = np.asarray(uh_dists_0_to_2[l])

        # condition 0 to 1
        uh_dists_1_to_2[l] = []
        for i in range(data_cond1[l].shape[0]):
            for j in range(data_cond2[l].shape[0]):
                traj1 = data_cond1[l][i,:].reshape((num_timesteps,-1))
                traj2 = data_cond2[l][j,:].reshape((num_timesteps,-1))
                dist_t = 1/num_neurons * np.sum(np.sqrt(((traj2-traj1)**2)), axis=1)
                uh_dists_1_to_2[l].append(dist_t)
        uh_dists_1_to_2[l] = np.asarray(uh_dists_1_to_2[l])

        if plot_indiv:
            plt.figure()

            # use standard error
            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_0_inner[l],axis=0), yerr=(np.std((uh_dists_0_inner[l]),axis=0) / np.sqrt(uh_dists_0_inner[l].shape[0])), color='r', label='tablet')
            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_1_inner[l],axis=0), yerr=(np.std((uh_dists_1_inner[l]),axis=0) / np.sqrt(uh_dists_1_inner[l].shape[0])), color='y', label='mechanical')
            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_2_inner[l],axis=0), yerr=(np.std((uh_dists_2_inner[l]),axis=0) / np.sqrt(uh_dists_2_inner[l].shape[0])), color='b', label='social')

            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_0_to_1[l],axis=0), yerr=(np.std((uh_dists_0_to_1[l]),axis=0) / np.sqrt(uh_dists_0_to_1[l].shape[0])), color='orange', linestyle='--', label='tablet-mechanical')
            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_0_to_2[l],axis=0), yerr=(np.std((uh_dists_0_to_2[l]),axis=0) / np.sqrt(uh_dists_0_to_2[l].shape[0])), color='purple', linestyle='--', label='tablet-social')
            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_1_to_2[l],axis=0), yerr=(np.std((uh_dists_1_to_2[l]),axis=0) / np.sqrt(uh_dists_1_to_2[l].shape[0])), color='g', linestyle='--', label='mechanical-social')

            """
            # use standard deviation
            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_0_inner[l],axis=0), yerr=(np.std((uh_dists_0_inner[l]),axis=0)), color='r', label='tablet')
            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_1_inner[l],axis=0), yerr=(np.std((uh_dists_1_inner[l]),axis=0)), color='y', label='mechanical')
            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_2_inner[l],axis=0), yerr=(np.std((uh_dists_2_inner[l]),axis=0)), color='b', label='social')

            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_0_to_1[l],axis=0), yerr=(np.std((uh_dists_0_to_1[l]),axis=0)), color='orange', linestyle='--', label='tablet-mechanical')
            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_0_to_2[l],axis=0), yerr=(np.std((uh_dists_0_to_2[l]),axis=0)), color='purple', linestyle='--', label='tablet-social')
            plt.errorbar(np.arange(num_timesteps), np.mean(uh_dists_1_to_2[l],axis=0), yerr=(np.std((uh_dists_1_to_2[l]),axis=0)), color='g', linestyle='--', label='mechanical-social')
            """

            plt.legend()
            plt.xlabel('time steps')
            plt.ylabel('average distance')
            plt.savefig(os.path.join(evaluation_folder, 'Distances_l-' + str(l) + filename))
            plt.close()

    plt.figure()
    plt.errorbar(np.arange(num_timesteps), np.mean(np.concatenate(uh_dists_0_inner),axis=0), yerr=(np.std((np.concatenate(uh_dists_0_inner)),axis=0) / np.sqrt(np.concatenate(uh_dists_0_inner).shape[0])), color='r', label='tablet')
    plt.errorbar(np.arange(num_timesteps), np.mean(np.concatenate(uh_dists_1_inner),axis=0), yerr=(np.std(np.concatenate(uh_dists_1_inner),axis=0) / np.sqrt(uh_dists_1_inner[l].shape[0])), color='y', label='mechanical')
    plt.errorbar(np.arange(num_timesteps), np.mean(np.concatenate(uh_dists_2_inner),axis=0), yerr=(np.std(np.concatenate(uh_dists_2_inner),axis=0) / np.sqrt(uh_dists_2_inner[l].shape[0])), color='b', label='social')

    plt.errorbar(np.arange(num_timesteps), np.mean(np.concatenate(uh_dists_0_to_1),axis=0), yerr=(np.std(np.concatenate(uh_dists_0_to_1),axis=0) / np.sqrt(uh_dists_0_to_1[l].shape[0])), color='orange', linestyle='--', label='tablet-mechanical')
    plt.errorbar(np.arange(num_timesteps), np.mean(np.concatenate(uh_dists_0_to_2),axis=0), yerr=(np.std(np.concatenate(uh_dists_0_to_2),axis=0) / np.sqrt(uh_dists_0_to_2[l].shape[0])), color='purple', linestyle='--', label='tablet-social')
    plt.errorbar(np.arange(num_timesteps), np.mean(np.concatenate(uh_dists_1_to_2),axis=0), yerr=(np.std(np.concatenate(uh_dists_1_to_2),axis=0) / np.sqrt(uh_dists_1_to_2[l].shape[0])), color='g', linestyle='--', label='mechanical-social')
    plt.legend()
    plt.xlabel('time steps')
    plt.ylabel('average distance')
    if ymax:
        plt.ylim([0, ymax])
    plt.savefig(os.path.join(evaluation_folder, 'Distances_all-lengths' + filename))
    plt.close()

    return uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2


# use data without any specific separation according to participants or lengths
def evaluate_activations_distances_across_categories(data_cond0, data_cond1, data_cond2, evaluation_folder, filename, num_neurons=25, num_timesteps=22, plot_indiv = False, ymax=None):

    data_cond0_concat = [np.concatenate(data_cond0)]
    data_cond1_concat = [np.concatenate(data_cond1)]
    data_cond2_concat = [np.concatenate(data_cond2)]

    uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2 = evaluate_activations_distances(data_cond0_concat, data_cond1_concat, data_cond2_concat, evaluation_folder, filename, num_neurons, num_timesteps, plot_indiv = False, ymax=ymax)

    return uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2


