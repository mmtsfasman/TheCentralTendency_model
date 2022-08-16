import os
from nets import load_network
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import seaborn as sns
import scipy
from utils.calc_regression_indices import calc_regression_indices
from utils.quantitative_distances import identify_lengths, sort_by_length_and_condition, sort_by_participant_and_condition, evaluate_activations, evaluate_activations_distances, evaluate_activations_distances_across_categories
from utils.plot_with_covariance import plot_with_covariance

colors = [(0.988235294, 0.552941176, 0.384313725), (0.4, 0.760784314, 0.647058824), (0.552941176, 0.62745098, 0.796078431)]
colors = ['r', '#f3a712', 'b']
markers = ['^', 's', 'd']
        
# evaluate the network behavior when setting the different initial states and modifying H_test
train_H_values = [1]
train_sigma_ext = 0.001
test_H_values = [1, 0.5, 0.4, 0.1, 0.05, 1000, 1e6, 1e9]

plot_all = False # create all evaluation plots
compte_activation_trace_quantification = True

plot_across_length_data = True # compare only the same participants to visualize the differences between the lengths
plot_across_participant_data = True # compare only the same lengths to visualize the differences between participants
plot_across_length_participant_data = False
plot_qbehavior = False
plot_qactivations = True
plot_qactivationspca = False

for trainRun, H_train in enumerate(train_H_values):
    print('H_train: ' + str(H_train))

    # the ten final networks for which analysis should be conducted, trained with H=1, sigma_ext = 0.001
    
    folders = [
        #'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2020-12-18_17-01_0997494/',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-21_13-38_0701178/',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-21_13-46_0242292/',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-21_13-47_0425773/',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-21_13-47_0470388/',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-21_13-48_0583639/',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-22_10-26_0684351/',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-22_10-27_0506050/',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-22_10-27_0591835/',
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2021-01-22_10-27_0679615/'
    ]
    
    for folder_id, f in enumerate(folders):
        network_to_load = f + str(H_train)
        print("Load network: " + network_to_load)
        params, train_model = load_network(network_to_load, model_filename='network-final.npz')
        num_conditions = 3
        num_timesteps = 22
        num_participants = 25
        subjects = range(num_participants)

        evaluation_folder = os.path.join(network_to_load, "evaluation")
        pathlib.Path(evaluation_folder).mkdir(parents=True, exist_ok=True)

        x_train_norm = np.float32(np.load('./human_data/presented_norm.npy'))
        x_human_norm = np.float32(np.load('./human_data/human_norm.npy'))
        classes_train = np.load('./human_data/classes.npy')
        cond_list = np.load('./human_data/conditions.npy')
        subj_list = np.load('./human_data/subjects.npy')

        presented_lengths = x_train_norm[:,-1] - x_train_norm[:,0]
        human_reproduced_lengths = x_human_norm[:,-1] - x_human_norm[:,0]

        # which initial states to use for the testing data
        initial_states_test = train_model.initial_states.W.array[classes_train,:]

        # feed the network the data
        for testRun, H in enumerate(test_H_values):
            print('H: ' + str(H))

            try:
                results = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_results.npy"))
                resultsv = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_resultsv.npy"))
                resultsm = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_resultsm.npy"))
                pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_pred_error.npy"))
                weighted_pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_weighted_pred_error.npy"))
                u_h_history = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_u_h_history.npy"))
                resultspos = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_resultspos.npy"))
                print('loaded existing results')
            except:
                print(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_resultsm.npy"))
                print('generate results')
                results, resultsv, resultsm, pred_error, weighted_pred_error, u_h_history, resultspos = train_model.generate(initial_states_test, num_timesteps, external_input = x_train_norm, add_variance_to_output = 0, additional_output='activations', hyp_prior = H)

                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_results.npy"), results)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_resultsv.npy"), resultsv)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_resultsm.npy"),  resultsm)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_pred_error.npy"), pred_error)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_weighted_pred_error.npy"), weighted_pred_error)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_u_h_history.npy"), u_h_history)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H) + "_resultspos.npy"), resultspos)

            reproduced_lengths = results[:,-1] - results[:,0]

            # calculate the regression indices from the model reproductions, and the human data
            model_RI, slope_model, intercept_model = calc_regression_indices(presented_lengths, reproduced_lengths, classes_train)
            human_RI, slope_human, intercept_human = calc_regression_indices(presented_lengths, human_reproduced_lengths, classes_train)

            if plot_all:
                # PLOT: MODEL RESULTS #
                #######################

                fig = plt.figure(figsize = (25,25))
                axObj = np.empty((num_participants,), dtype=object)
                # plot identity lines for individual participants to check that everything is correct
                for i in range(num_participants):
                    axObj[i] = fig.add_subplot(5,5,i+1)
                    axObj[i].plot([0.6, 1.7], [0.6, 1.7])

                for i,is_id in enumerate(classes_train):
                    current_cond = cond_list[i] # is_id%num_conditions
                    current_subj = subj_list[i] # int((is_id - is_id%num_conditions)/num_conditions)
                    #print("{} (condition: {}, subject: {})".format(i, current_cond, current_subj))

                    axObj[current_subj].scatter(presented_lengths[i], reproduced_lengths[i], marker='*', color=colors[current_cond])
                plt.savefig(os.path.join(evaluation_folder, "H-" + str(H) + "_model-results.png"))
                print("plotted model results: " + "H-" + str(H) + "_model-results.png")

                # PLOT: human data for verification #
                #####################################
                
                fig = plt.figure(figsize = (25,25))
                parameters = {'xtick.labelsize': 25,
                              'ytick.labelsize': 25}
                plt.rcParams.update(parameters)
                axObj = np.empty((num_participants,), dtype=object)

                xmin = np.min(presented_lengths)
                xmax = np.max(presented_lengths)
                ymin = np.min(human_reproduced_lengths)
                ymax = np.max(human_reproduced_lengths)
                xymin = min([xmin,ymin])
                xymax = max([xmax,ymax])

                # plot identity lines for individual participants to check that everything is correct
                for i in range(num_participants):
                    axObj[i] = fig.add_subplot(5,5,i+1)
                    axObj[i].plot([xymin, xymax], [xymin, xymax], 'k', linewidth=3)

                for i,is_id in enumerate(classes_train):
                    current_cond = cond_list[i] # is_id%num_conditions
                    current_subj = subj_list[i] # int((is_id - is_id%num_conditions)/num_conditions)
                    
                    axObj[current_subj].scatter(presented_lengths[i], human_reproduced_lengths[i], marker=markers[current_cond], color=colors[current_cond])
                    axObj[current_subj].set_xlim([xmin,xmax])
                    axObj[current_subj].set_ylim([ymin,ymax])
                    
                for is_id in range(num_participants*num_conditions):
                    current_cond = is_id%num_conditions
                    current_subj = int((is_id - is_id%num_conditions)/num_conditions)
                    #print("{} (condition: {}, subject: {})".format(is_id, current_cond, current_subj))

                    x = np.arange(xmin, xmax, 0.1)
                    #y = slope_model[is_id]*x+intercept_model[is_id]
                    y_human = slope_human[is_id]*x+intercept_human[is_id]
                    #axObj[current_subj].plot(x, y, "k")
                    axObj[current_subj].plot(x, y_human, color=colors[current_cond], linewidth=3)
                    axObj[current_subj].set_xlim([xmin,xmax])
                    axObj[current_subj].set_ylim([ymin,ymax])
                    
                plt.savefig('verification-human-data.pdf')
                print("plotted human data for verification: verification-human-data.pdf")


                # PLOT: Model results + human regression #
                ##########################################
                xmin = np.min(presented_lengths)
                xmax = np.max(presented_lengths)
                ymin = np.min(human_reproduced_lengths)
                ymax = np.max(human_reproduced_lengths)
                xymin = min([xmin,ymin])
                xymax = max([xmax,ymax])
                
                fig = plt.figure(figsize = (25,25))
                parameters = {'xtick.labelsize': 25,
                              'ytick.labelsize': 25}
                plt.rcParams.update(parameters)
                axObj = np.empty((num_participants,), dtype=object)
                for i in range(num_participants):
                    axObj[i] = fig.add_subplot(5,5,i+1)
                    axObj[i].plot([xymin, xymax], [xymin,xymax], 'k', linewidth=3)

                for i,is_id in enumerate(classes_train):
                    current_cond = cond_list[i] # is_id%num_conditions
                    current_subj = subj_list[i] # int((is_id - is_id%num_conditions)/num_conditions)
                    #print("{} (condition: {}, subject: {})".format(i, current_cond, current_subj))

                    axObj[current_subj].scatter(presented_lengths[i], reproduced_lengths[i], marker=markers[current_cond], color=colors[current_cond])

                for is_id in range(num_participants*num_conditions):
                    current_cond = is_id%num_conditions
                    current_subj = int((is_id - is_id%num_conditions)/num_conditions)
                    #print("{} (condition: {}, subject: {})".format(is_id, current_cond, current_subj))

                    x = np.arange(xmin, xmax, 0.1)
                    y = slope_model[is_id]*x+intercept_model[is_id]
                    #y_human = slope_human[is_id]*x+intercept_human[is_id]
                    axObj[current_subj].plot(x, y, color=colors[current_cond], linewidth=3)
                    #axObj[current_subj].plot(x, y_human, color=colors[current_cond])
                    axObj[current_subj].set_xlim([xmin,xmax])
                    axObj[current_subj].set_ylim([ymin,ymax])

                plt.savefig(os.path.join(evaluation_folder, "H-" + str(H) + "_model-results-with-RI.pdf"))
                plt.close()
                print("plotted model results with human regression: " + "H-" + str(H) + "_model-results-with-RI.pdf")


                # PLOT: human data + human regression #
                #######################################

                fig = plt.figure(figsize = (25,25))
                axObj = np.empty((num_participants,), dtype=object)
                for i in range(num_participants):
                    axObj[i] = fig.add_subplot(5,5,i+1)
                    axObj[i].plot([0.6, 1.7], [0.6, 1.7])

                for i,is_id in enumerate(classes_train):
                    current_cond = is_id%num_conditions
                    current_subj = int((is_id - is_id%num_conditions)/num_conditions)
                    #print("{} (condition: {}, subject: {})".format(i, current_cond, current_subj))
                    axObj[current_subj].scatter(presented_lengths[i], human_reproduced_lengths[i], marker='*', color=colors[current_cond])

                for is_id in range(num_participants*num_conditions):
                    current_cond = is_id%num_conditions
                    current_subj = int((is_id - is_id%num_conditions)/num_conditions)
                    print("{} (condition: {}, subject: {})".format(is_id, current_cond, current_subj))

                    x = np.arange(np.min(presented_lengths), np.max(presented_lengths), 0.1)
                    y = slope_human[is_id]*x+intercept_human[is_id]
                    axObj[current_subj].plot(x, y, "g")

                plt.savefig(os.path.join(evaluation_folder, "H-" + str(H) + "_human-data-with-RI.png"))
                plt.close()
                print("plotted human results with human regression: " + "H-" + str(H) + "_human-data-with-RI.png")

            # PLOT: regression index comparison human–model #
            #################################################

            fig = plt.figure(figsize=(8,7))
            parameters = {'xtick.labelsize': 24,
                      'ytick.labelsize': 24,
                      'axes.labelsize': 24,
                      'axes.titlesize': 30}
            plt.rcParams.update(parameters)
            plt.plot([np.min([model_RI, human_RI])-0.05, np.max([model_RI, human_RI])+0.05], [np.min([model_RI, human_RI])-0.05, np.max([model_RI, human_RI])+0.05], 'k', linewidth=3)
            for i in range(num_conditions*num_participants):
                current_cond = i%num_conditions
                plt.plot(model_RI[i], human_RI[i], marker=markers[current_cond], color=colors[current_cond]);
            plt.xlabel('model RI')
            plt.ylabel('human RI')
            plt.xlim([np.min([model_RI, human_RI])-0.05, np.max([model_RI, human_RI])+0.05])
            plt.ylim([np.min([model_RI, human_RI])-0.05, np.max([model_RI, human_RI])+0.05])
            plt.savefig(os.path.join(evaluation_folder, "H-" + str(H) + "_human-vs-model-RI.pdf"))
            plt.close()

            # write into csv file
            with open(os.path.join(evaluation_folder, "H-" + str(H) + "_human-vs-model-RI.csv"), 'w') as f:
                f.write("condition\tparticipant\thuman_RI\tmodel_RI\n")
                for is_id in range(num_conditions*num_participants):
                    current_cond = i%num_conditions
                    current_subj = int((is_id - is_id%num_conditions)/num_conditions)
                    f.write(str(current_cond) + "\t" + str(current_subj) + "\t" + str(human_RI[i]) + "\t" + str(model_RI[i]) + "\n")
                    
            np.save(os.path.join(evaluation_folder,"H-" + str(H) + "_modelRI.npy"), model_RI)
            np.save(os.path.join(evaluation_folder,"H-" + str(H) + "_humanRI.npy"), human_RI)

            print("regression index comparison human-model" + "H-" + str(H) + "_modelRI.npy" + " and " + "H-" + str(H) + "_humanRI.npy")

            # plot mechanical vs. social condition trend #
            ##############################################
            plt.figure()
            plt.plot(human_RI[1:75:3], human_RI[2:75:3], 'k*')
            plt.plot(model_RI[1:75:3], model_RI[2:75:3], 'm*')
            plt.xlabel('mechanical')
            plt.ylabel('social')
            plt.savefig(os.path.join(evaluation_folder, 'H-' + str(H) + '_verification-mechanical-social-human-data.png'))
            print("plotted mechanical vs. social condition trend" + 'H-' + str(H) + '_verification-mechanical-social-human-data.png')

            ##########################
            # INITIAL STATES AND PCA #
            ##########################

            # Generate PCA #
            ################

            # PCA of internal representations: num_samples x (num_timesteps * num_neurons)
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # collect all data in all time steps
            all_activations = u_h_history.reshape((u_h_history.shape[0]*num_timesteps,-1))
            all_activations = np.concatenate((all_activations, train_model.initial_states.W.array),axis=0)
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

            plt.savefig(os.path.join(evaluation_folder, "H-" + str(H) + "_pca-per-participant-results.png"))
            plt.close()
            print("plotted initial states per participant: " + "H-" + str(H) + "_pca-per-participant-results.png")

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

            plt.savefig(os.path.join(evaluation_folder, "H-" + str(H) + "_pca-per-condition-results.png"))
            plt.close()
            print("plotted initial states per condition: " + "H-" + str(H) + "_pca-per-condition-results.png")

            # plot the initial states
            initial_states_pca = pca.transform(scaler.transform(train_model.initial_states.W.array))
            
            plot_with_covariance(initial_states_pca, np.tile(np.asarray([0,1,2]), (25,)), os.path.join(evaluation_folder, "H-" + str(H) + "_initial-states-results.pdf"))
            # plot first steps in the generation only
            plot_with_covariance(first_step_pca, cond_list, os.path.join(evaluation_folder, "H-" + str(H) + "_first-time-step-results.pdf"))
            # plot last steps in the generation only
            plot_with_covariance(last_step_pca, cond_list, os.path.join(evaluation_folder, "H-" + str(H) + "_last-time-step-results.pdf"))
                    
            print("******************************************\nPCA explained variance: "
                + str(np.sum(pca.explained_variance_ratio_))
                + "\n******************************************\n")
            
            """
            colors = sns.color_palette('husl', n_colors=num_participants)
            # plot first steps in the generation only
            plt.figure(figsize = (5,5))
            for i,is_id in enumerate(classes_train):
                current_cond = cond_list[i] # is_id%num_conditions
                current_subj = subj_list[i] # int((is_id - is_id%num_conditions)/num_conditions)
                plt.scatter(first_step_pca[i,0], first_step_pca[i,1], marker='*', color=colors[current_subj])
            plt.savefig(os.path.join(evaluation_folder, "H-" + str(H) + "_first-time-step-results_subjs.png"))
            plt.close()

            # plot last steps in the generation only
            plt.figure(figsize = (5,5))
            for i,is_id in enumerate(classes_train):
                current_cond = cond_list[i] # is_id%num_conditions
                current_subj = subj_list[i] # int((is_id - is_id%num_conditions)/num_conditions)
                plt.scatter(last_step_pca[i,0], last_step_pca[i,1], marker='*', color=colors[current_subj])
            plt.savefig(os.path.join(evaluation_folder, "H-" + str(H) + "_last-time-step-results_subjs.png"))
            plt.close()
            """


            if compte_activation_trace_quantification:
                # sort data by presented length #
                #################################
                valid_trials, presented_lengths_valid_main, main_lengths, [u_h_history_split, classes_train_split, results_split, cond_list_split, subj_list_split] = identify_lengths(presented_lengths, [u_h_history, classes_train, results, cond_list, subj_list])

                # 11 x (N x (70*22)), N = number of length reproductions in this bin
                full_data_per_length_cond0, full_data_per_length_cond1, full_data_per_length_cond2 = sort_by_length_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)

                # 11 x (N x (22))
                results_per_length_cond0, results_per_length_cond1, results_per_length_cond2 = sort_by_length_and_condition(results_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)

                # 11 x (N x (2*22)), 2 = PCA-dim, N = number of length reproductions in this bin
                pca_data_per_length_cond0, pca_data_per_length_cond1, pca_data_per_length_cond2 = sort_by_length_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22, scaler=scaler, pca=pca)

            
                # Neural activation traces: quantitative distances #
                ####################################################

                # within-lengths, across-participants
                print("Compute neural activation traces, quantitative distances")
                if plot_across_participant_data:
                    if  plot_qactivations:
                        uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2 = evaluate_activations_distances(full_data_per_length_cond0, full_data_per_length_cond1, full_data_per_length_cond2, evaluation_folder, "_activations_H-" + str(H), train_model.num_c, plot_indiv = False, ymax=0.5)
                        np.save(os.path.join(evaluation_folder, 'activations_within-lengths_across-participants_H-' + str(H) + '.npy'), [uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2])

                    if plot_qbehavior:
                        uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2 = evaluate_activations_distances(results_per_length_cond0, results_per_length_cond1, results_per_length_cond2, evaluation_folder, "_behavior_H-" + str(H), train_model.num_c, plot_indiv = False, ymax=0.02)
                        np.save(os.path.join(evaluation_folder, 'behavior_within-lengths_across-participants_H-' + str(H) + '.npy'), [uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2])

                    if plot_qactivationspca:
                        uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2 = evaluate_activations_distances(pca_data_per_length_cond0, pca_data_per_length_cond1, pca_data_per_length_cond2, evaluation_folder, "_pca-activations_H-" + str(H), train_model.num_c, plot_indiv = False, ymax=0.5)


                if plot_across_length_participant_data:
                    # across-lengths, across-participants, just compare everything with everything without any sorting
                    if plot_qactivations:
                        uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2 = evaluate_activations_distances_across_categories(full_data_per_length_cond0, full_data_per_length_cond1, full_data_per_length_cond2, evaluation_folder, "_activations_H-" + str(H) + "_across-lengths", train_model.num_c, plot_indiv = False, ymax=0.5)
                        np.save(os.path.join(evaluation_folder, 'activations_across-lengths_across-participants_H-' + str(H) + '.npy'), [uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2])
                    if plot_qbehavior:
                        uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2 = evaluate_activations_distances_across_categories(results_per_length_cond0, results_per_length_cond1, results_per_length_cond2, evaluation_folder, "_behavior_H-" + str(H) + "_across-lengths", train_model.num_c, plot_indiv = False, ymax=0.02)
                        np.save(os.path.join(evaluation_folder, 'behavior_across-lengths_across-participants_H-' + str(H) + '.npy'), [uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2])
                    if plot_qactivationspca:
                        uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2 = evaluate_activations_distances_across_categories(pca_data_per_length_cond0, pca_data_per_length_cond1, pca_data_per_length_cond2, evaluation_folder, "_pca-activations_H-" + str(H) + "_across-lengths", train_model.num_c, plot_indiv = False, ymax=0.5)


                    # within-participants, across lengths
                if plot_across_length_data:
                    # get data split by condition and by participant
                    full_data_per_p_cond0, full_data_per_p_cond1, full_data_per_p_cond2 = sort_by_participant_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                    results_per_p_cond0, results_per_p_cond1, results_per_p_cond2 = sort_by_participant_and_condition(results_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                    pca_data_per_p_cond0, pca_data_per_p_cond1, pca_data_per_p_cond2 = sort_by_participant_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22, scaler=scaler, pca=pca)

                    if plot_qactivations:
                        uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2 = evaluate_activations_distances(full_data_per_p_cond0, full_data_per_p_cond1, full_data_per_p_cond2, evaluation_folder, "_activations_H-" + str(H) + "_per-participant", train_model.num_c, plot_indiv = False, ymax=0.5)
                        np.save(os.path.join(evaluation_folder, 'activations_across-lengths_within-participants_H-' + str(H) + '.npy'), [uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2])

                        #evaluate_activations_distances_across_categories(full_data_per_p_cond0, full_data_per_p_cond1, full_data_per_p_cond2, evaluation_folder, "_activations_H-" + str(H) + "_across-participants-should-equal-across_lengths", train_model.num_c, plot_indiv = False, ymax=0.5)
                    if plot_qbehavior:
                        uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2 = evaluate_activations_distances(pca_data_per_p_cond0, pca_data_per_p_cond1, pca_data_per_p_cond2, evaluation_folder, "_pca-activations_H-" + str(H) + "_per-participant", train_model.num_c, plot_indiv = False, ymax=0.02)
                        np.save(os.path.join(evaluation_folder, 'behavior_across-lengths_within-participants_H-' + str(H) + '.npy'), [uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2])

                    if plot_qactivationspca:
                        uh_dists_0_inner, uh_dists_1_inner, uh_dists_2_inner, uh_dists_0_to_1, uh_dists_0_to_2, uh_dists_1_to_2 = evaluate_activations_distances(results_per_p_cond0, results_per_p_cond1, results_per_p_cond2, evaluation_folder, "_behavior_H-" + str(H) + "_per-participant", train_model.num_c, plot_indiv = False, ymax=0.5)

                print("Saved to: activations_across-lengths...npy/behavior_across_lengths...")

"""
fig = plt.figure()
for i in range(11):
    ax = fig.add_subplot(2,6,1+i)
    for j in range(results_per_length_cond0[i].shape[0]):
        traj = results_per_length_cond0[i][j,:].reshape((22,-1))
        ax.plot(np.arange(len(traj)), traj, 'r')
        ax.plot([21], traj[-1], 'b*')
plt.show()

for i in range(11):
    ax = fig.add_subplot(11,3,2)
    for j in range(pca_data_per_length_cond1[i].shape[0]):
        traj = pca_data_per_length_cond1[i][j,:].reshape((22,2))
        ax.plot(traj[:,0], traj[:,1], 'y')
ax = fig.add_subplot(11,3,3)
for i in range(11):
    for j in range(pca_data_per_length_cond2[i].shape[0]):
        traj = pca_data_per_length_cond2[i][j,:].reshape((22,2))
        ax.plot(traj[:,0], traj[:,1], 'b')
plt.show()


plt.figure()
rawresults_tab = results[np.where(cond_list==0)]
for i in range(rawresults_tab.shape[0]):
    #plt.plot(np.arange(22), rawresults_tab[i,:], 'r')
    plt.plot([21], rawresults_tab[i,-1]), 'b*'
plt.show()

# select only one participant and one condition
rawresults_tab = results[np.where(cond_list==1)]
corresp_subj = subj_list[np.where(cond_list==1)]
only_one_partic_tab = rawresults_tab[np.where(corresp_subj==0)]
np.var(only_one_partic_tab[:,-1])

"""

