# Generate and plot the behavioral data (RI) for the trained H and for aberrant H values

import os
import pathlib

from nets import load_network
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import scipy
from utils.calc_regression_indices import calc_regression_indices
from utils.boxplot_fct import boxplot_fct
from utils.quantitative_distances import identify_lengths, sort_by_length_and_condition, sort_by_participant_and_condition, evaluate_activations_distances, evaluate_activations

modify_prior = True
modify_sigma_inp = True

add_variance_to_output = 0 #None
override = False
# evaluate what is plotted with different initial states

x_train_norm = np.float32(np.load('./human_data/presented_norm.npy'))
x_human_norm = np.float32(np.load('./human_data/human_norm.npy'))
classes_train = np.load('./human_data/classes.npy')
cond_list = np.load('./human_data/conditions.npy')
subj_list = np.load('./human_data/subjects.npy')

folders = [
        'results/training/all_human/human_training_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_2020-12-18_17-01_0997494/',
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

hypo_H_values = [1]#, 1000, 1e6, 1e9]
hyper_H_values = [1, 0.5, 0.4, 0.1, 0.05]#[1, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
pathlib.Path('results/network_summary/').mkdir(parents=True, exist_ok=True)
np.save('./results/network_summary/hyper_H_values.npy', hyper_H_values)
np.save('./results/network_summary/hypo_H_values.npy', hypo_H_values)

for folder_id, network_to_load in enumerate(folders):
    # H and sigma values that were used for training
    train_H_values = [1]
    train_sigma_ext = [1e-3]
    
    for trainRun, H_train in enumerate(train_H_values):
        print('H_train: ' + str(H_train))
        network_to_load_folder = os.path.join(network_to_load, str(H_train))
        print(network_to_load)
        evaluation_folder = os.path.join(network_to_load_folder, "evaluation")
        pathlib.Path(evaluation_folder).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(evaluation_folder, 'hyper_H_values.npy'), hyper_H_values)
        np.save(os.path.join(evaluation_folder, 'hypo_H_values.npy'), hypo_H_values)

        increase_sigma_inp = [H_train / (i * H_train / train_sigma_ext[trainRun]) for i in hyper_H_values]
        decrease_sigma_inp = [H_train / (i * H_train / train_sigma_ext[trainRun]) for i in hypo_H_values]
        np.save(os.path.join(evaluation_folder, 'increase_sigma_inp.npy'), increase_sigma_inp)
        np.save(os.path.join(evaluation_folder, 'decrease_sigma_inp.npy'), decrease_sigma_inp)

        if modify_prior:
            try:
                params, train_model = load_network(network_to_load_folder, model_filename='network-final.npz')
            except FileNotFoundError:
                print("Network file not found in: " + str(network_to_load_folder) + "\nContinue.")
                continue

            diff_H_sigma = train_H_values[trainRun] - train_sigma_ext[trainRun]
            
            num_conditions = 3
            num_timesteps = 22
            num_participants = 25
            subjects = range(num_participants)

            initial_states_test = train_model.initial_states.W.array[classes_train,:]
            
            # regression indices of human and model
            try:
                if override:
                    raise FileNotFoundError('Force recreation of files')
                results = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_results.npy"))
                resultsv = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultsv.npy"))
                resultsm = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultsm.npy"))
                pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_pred_error.npy"))
                weighted_pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_weighted_pred_error.npy"))
                u_h_history = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_u_h_history.npy"))
                resultspos = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultspos.npy"))
                print('loaded existing')
            except FileNotFoundError:
                print('File not found: ' + os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_results.npy"))
                print('generate')
                results, resultsv, resultsm, pred_error, weighted_pred_error, u_h_history, resultspos = train_model.generate(initial_states_test, num_timesteps, external_input = x_train_norm, add_variance_to_output = add_variance_to_output, additional_output='activations', hyp_prior = H_train)
                
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_results.npy"), results)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultsv.npy"), resultsv)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultsm.npy"),  resultsm)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_pred_error.npy"), pred_error)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_weighted_pred_error.npy"), weighted_pred_error)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_u_h_history.npy"), u_h_history)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultspos.npy"), resultspos)


            presented_lengths = x_train_norm[:,-1] - x_train_norm[:,0]
            reproduced_lengths = results[:,-1] - results[:,0]
            human_reproduced_lengths = x_human_norm[:,-1] - x_human_norm[:,0]

            # calculate the regression indices from the model reproductions, and the human data
            model_RI, slope_model, intercept_model = calc_regression_indices(presented_lengths, reproduced_lengths, classes_train)
            human_RI, slope_human, intercept_human = calc_regression_indices(presented_lengths, human_reproduced_lengths, classes_train)
            
            # regression index for the different conditions
           
            RI_tablet_h = human_RI[0:num_participants*num_conditions:num_conditions]
            RI_mech_h = human_RI[1:num_participants*num_conditions:num_conditions]
            RI_soc_h = human_RI[2:num_participants*num_conditions:num_conditions]
            RI_tablet_m = model_RI[0:num_participants*num_conditions:num_conditions]
            RI_mech_m = model_RI[1:num_participants*num_conditions:num_conditions]
            RI_soc_m = model_RI[2:num_participants*num_conditions:num_conditions]

            plt.figure()
            plt.plot(RI_mech_m, RI_soc_m, '*')
            plt.xlabel('mechanical')
            plt.ylabel('social')
            plt.savefig(os.path.join(evaluation_folder, 'RI-mech-soc_model.png'))
            plt.close()
            plt.figure()
            plt.plot(RI_mech_h, RI_soc_h, '*')
            plt.xlabel('mechanical')
            plt.ylabel('social')
            plt.savefig(os.path.join(evaluation_folder, 'RI-mech-soc_human.png'))
            plt.close()

            human_diff_tablet_mechanical = RI_tablet_h - RI_mech_h
            human_diff_tablet_social = RI_tablet_h - RI_soc_h
            human_diff_mechanical_social = RI_mech_h - RI_soc_h
            model_diff_tablet_mechanical = RI_tablet_m - RI_mech_m
            model_diff_tablet_social = RI_tablet_m - RI_soc_m
            model_diff_mechanical_social = RI_mech_m - RI_soc_m

            boxplot_fct([human_diff_tablet_mechanical, model_diff_tablet_mechanical, human_diff_tablet_social, model_diff_tablet_social, human_diff_mechanical_social, model_diff_mechanical_social], os.path.join(evaluation_folder, 'human-model_comparison.pdf'), colors=['k','m','k','m','k', 'm'], labels=['Tablet-Mechanical', '', 'Tablet-Social', '', 'Mechanical-Social', ''], xlabel="", title="", figsize=(10,7))#, ylim=[-0.5,0.6])

            np.save('results/human_RI_tablet_vs_mechanical.npy', human_diff_tablet_mechanical)
            np.save('results/human_RI_tablet_vs_social.npy', human_diff_tablet_social)
            np.save('results/human_RI_mechanical_vs_social.npy', human_diff_mechanical_social)
            np.save('results/model_RI_tablet_vs_mechanical_net-' + str(folder_id) + '.npy', model_diff_tablet_mechanical)
            np.save('results/model_RI_tablet_vs_social_net-' + str(folder_id) + '.npy', model_diff_tablet_social)
            np.save('results/model_RI_mechanical_vs_social_net-' + str(folder_id) + '.npy', model_diff_mechanical_social)

            """
            plt.figure()
            plt.plot(np.tile(1, (25,)), model_diff_tablet_mechanical, '*')
            plt.plot(np.tile(2, (25,)), human_diff_tablet_mechanical, '*')
            plt.plot(np.tile(3, (25,)), model_diff_tablet_social, '*')
            plt.plot(np.tile(4, (25,)), human_diff_tablet_social, '*')
            plt.plot(np.tile(5, (25,)), model_diff_mechanical_social, '*')
            plt.plot(np.tile(6, (25,)), human_diff_mechanical_social, '*')
            plt.savefig(os.path.join(evaluation_folder, 'model-human_comparison.png'))
            plt.close()

            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            median_value = np.percentile(model_diff_tablet_mechanical, 50)
            upper_quantile = np.percentile(model_diff_tablet_mechanical, 90)
            lower_quantile = np.percentile(model_diff_tablet_mechanical, 10)
            lower_std_dev = np.mean(model_diff_tablet_mechanical) - np.sqrt(np.var(model_diff_tablet_mechanical))
            upper_std_dev = np.mean(model_diff_tablet_mechanical) + np.sqrt(np.var(model_diff_tablet_mechanical))
            ax.boxplot([lower_std_dev, lower_quantile, median_value, upper_quantile, upper_std_dev], positions=[6])
            ax.scatter(np.tile(6, (25,)), model_diff_tablet_mechanical)
            """
            
            #plt.savefig(os.path.join(evaluation_folder, 'model-human_comparison.png'))
            
            # Use initial states for tablet and replicate mechanical and social #
            #####################################################################
            # difference tablet IS, H / tablet IS, moderate hypo H == difference tablet / mechanical
            # difference tablet IS, H / tablet IS, strong hypo H == difference tablet / social

            hypo_RIs = np.empty((len(hypo_H_values),),dtype=object)
            RI_tablet_m_hypo = np.empty((len(hypo_H_values),),dtype=object)
            RI_mech_m_hypo = np.empty((len(hypo_H_values),),dtype=object)
            RI_soc_m_hypo = np.empty((len(hypo_H_values),),dtype=object)

            for n, H_test in enumerate(hypo_H_values):
                #H_test = H_train * H_test_factor
                
                print('H: ' + str(H_test))
                try:
                    if override:
                        raise FileNotFoundError('Force recreation of files')
                    results = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_results.npy"))
                    resultsv = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultsv.npy"))
                    resultsm = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultsm.npy"))
                    pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_pred_error.npy"))
                    weighted_pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_weighted_pred_error.npy"))
                    u_h_history = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_u_h_history.npy"))
                    resultspos = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultspos.npy"))
                    print('loaded existing')
                except FileNotFoundError:
                    print('generate')
                    results, resultsv, resultsm, pred_error, weighted_pred_error, u_h_history, resultspos = train_model.generate(initial_states_test, num_timesteps, external_input = x_train_norm, add_variance_to_output = add_variance_to_output, additional_output='activations', hyp_prior = H_test)
                    
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_results.npy"), results)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultsv.npy"), resultsv)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultsm.npy"),  resultsm)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_pred_error.npy"), pred_error)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_weighted_pred_error.npy"), weighted_pred_error)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_u_h_history.npy"), u_h_history)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultspos.npy"), resultspos)
            
                reproduced_lengths = results[:,-1] - results[:,0]

                # calculate the regression indices from the model reproductions, and the human data
                model_RI, slope_model, intercept_model = calc_regression_indices(presented_lengths, reproduced_lengths, classes_train)
                hypo_RIs[n] = model_RI
                RI_tablet_m_hypo[n] = model_RI[0:num_participants*num_conditions:num_conditions]
                RI_mech_m_hypo[n] = model_RI[1:num_participants*num_conditions:num_conditions]
                RI_soc_m_hypo[n] = model_RI[2:num_participants*num_conditions:num_conditions]
            
                # plot the internal representations
                valid_trials, presented_lengths_valid_main, main_lengths, [u_h_history_split, classes_train_split, results_split, cond_list_split, subj_list_split] = identify_lengths(presented_lengths, [u_h_history, classes_train, results, cond_list, subj_list])

                # behavior
                #results_per_length_cond0, results_per_length_cond1, results_per_length_cond2 = sort_by_length_and_condition(results_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                #evaluate_activations_distances(results_per_length_cond0, results_per_length_cond1, results_per_length_cond2, evaluation_folder, "_behavior_H-" + str(H_test), train_model.num_c, plot_indiv = False, ymax=0.5)

                # activations
                full_data_per_length_cond0, full_data_per_length_cond1, full_data_per_length_cond2 = sort_by_length_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                evaluate_activations_distances(full_data_per_length_cond0, full_data_per_length_cond1, full_data_per_length_cond2, evaluation_folder, "_activations_H-" + str(H_test), train_model.num_c, plot_indiv = False, ymax=0.5)
                #pca, scaler = evaluate_activations(u_h_history, initial_states_test, cond_list, subj_list, classes_train, evaluation_folder, "modify-H_H-" + str(H_test), num_timesteps, num_participants)
                
                
    # get data split by condition and by participant
                full_data_per_p_cond0, full_data_per_p_cond1, full_data_per_p_cond2 = sort_by_participant_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                evaluate_activations_distances(full_data_per_p_cond0, full_data_per_p_cond1, full_data_per_p_cond2, evaluation_folder, "_activations_H-" + str(H_test) + "_per-participant", train_model.num_c, plot_indiv = False, ymax=0.5)
            
            """
            plt.figure()
            plt.plot(np.tile(0, (25,)), RI_tablet_m, '*')
            plt.plot(np.tile(5, (25,)), RI_mech_m, '*')
            plt.plot(np.tile(10, (25,)), RI_soc_m, '*')
            for n, H_test_factor in enumerate(hypo_H_values):
                plt.plot(np.tile(n+1, (25,)), RI_tablet_m_hypo[n], '*')
                plt.plot(np.tile(5+n+1, (25,)), RI_mech_m_hypo[n], '*')
                plt.plot(np.tile(10+n+1, (25,)), RI_soc_m_hypo[n], '*')
            plt.savefig(os.path.join(evaluation_folder, 'RI-change_hypo.png'))
            plt.close()
            """
            
            # RI_tablet_h ~ hyper   ~ default
            # RI_mech_h   ~ 
            # RI_soc_h    ~ hypo
            
            
            data = [RI_tablet_h - RI_mech_h, RI_tablet_h - RI_soc_h]
            labels = ["Human:T-M", "Human:T-S"]
            for n, H_test in enumerate(hypo_H_values):
                data.append(RI_tablet_m - RI_tablet_m_hypo[n])
                labels.append(str(H_test))
            boxplot_fct(data, os.path.join(evaluation_folder, 'modify-prior_tablet-hypo.pdf'), labels, title="Network with training prior reliance (H): " + str(H_train), figsize=(15,10))
            np.save(os.path.join(evaluation_folder, "modify-prior_tablet-hypo.npy"), data)
            
            # Use initial states for social and replicate tablet and mechanical #
            #####################################################################
            # difference social IS, H / social IS, moderate hyper H == difference social / mechanical
            # difference social IS, H / social IS, strong hyper H == difference social / tablet
            hyper_RIs = np.empty((len(hyper_H_values),),dtype=object)
            RI_tablet_m_hyper = np.empty((len(hyper_H_values),),dtype=object)
            RI_mech_m_hyper = np.empty((len(hyper_H_values),),dtype=object)
            RI_soc_m_hyper = np.empty((len(hyper_H_values),),dtype=object)

            for n, H_test in enumerate(hyper_H_values):
                #H_test = H_train * H_test_factor
            
                print('H: ' + str(H_test))
                try:
                    results = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_results.npy"))
                    resultsv = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultsv.npy"))
                    resultsm = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultsm.npy"))
                    pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_pred_error.npy"))
                    weighted_pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_weighted_pred_error.npy"))
                    u_h_history = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_u_h_history.npy"))
                    resultspos = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultspos.npy"))
                    print('loaded existing')
                except FileNotFoundError:
                    print('generate')
                    results, resultsv, resultsm, pred_error, weighted_pred_error, u_h_history, resultspos = train_model.generate(initial_states_test, num_timesteps, external_input = x_train_norm, add_variance_to_output = add_variance_to_output, additional_output='activations', hyp_prior = H_test)
                    
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_results.npy"), results)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultsv.npy"), resultsv)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultsm.npy"),  resultsm)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_pred_error.npy"), pred_error)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_weighted_pred_error.npy"), weighted_pred_error)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_u_h_history.npy"), u_h_history)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_test) + "_resultspos.npy"), resultspos)
                
                reproduced_lengths = results[:,-1] - results[:,0]

                valid_trials, presented_lengths_valid_main, main_lengths, [u_h_history_split, classes_train_split, results_split, cond_list_split, subj_list_split] = identify_lengths(presented_lengths, [u_h_history, classes_train, results, cond_list, subj_list])

                # behavior
                #results_per_length_cond0, results_per_length_cond1, results_per_length_cond2 = sort_by_length_and_condition(results_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                #evaluate_activations_distances(results_per_length_cond0, results_per_length_cond1, results_per_length_cond2, evaluation_folder, "_behavior_H-" + str(H_test), train_model.num_c, plot_indiv = False, ymax=0.5)

                # activations
                full_data_per_length_cond0, full_data_per_length_cond1, full_data_per_length_cond2 = sort_by_length_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                evaluate_activations_distances(full_data_per_length_cond0, full_data_per_length_cond1, full_data_per_length_cond2, evaluation_folder, "_activations_H-" + str(H_test), train_model.num_c, plot_indiv = False, ymax=0.5)
                #pca, scaler = evaluate_activations(u_h_history, initial_states_test, cond_list, subj_list, classes_train, evaluation_folder, "modify-H_H-" + str(H_test), num_timesteps, num_participants, plot_indiv = False, ymax=0.5)

    # get data split by condition and by participant
                full_data_per_p_cond0, full_data_per_p_cond1, full_data_per_p_cond2 = sort_by_participant_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                evaluate_activations_distances(full_data_per_p_cond0, full_data_per_p_cond1, full_data_per_p_cond2, evaluation_folder, "_activations_H-" + str(H_test) + "_per-participant", train_model.num_c, plot_indiv = False, ymax=0.5)

                # calculate the regression indices from the model reproductions, and the human data
                model_RI, slope_model, intercept_model = calc_regression_indices(presented_lengths, reproduced_lengths, classes_train)
                hyper_RIs[n] = model_RI
                RI_tablet_m_hyper[n] = model_RI[0:num_participants*num_conditions:num_conditions]
                RI_mech_m_hyper[n] = model_RI[1:num_participants*num_conditions:num_conditions]
                RI_soc_m_hyper[n] = model_RI[2:num_participants*num_conditions:num_conditions]
            """
            plt.figure()
            plt.plot(np.tile(0, (25,)), RI_tablet_m, '*')
            plt.plot(np.tile(5, (25,)), RI_mech_m, '*')
            plt.plot(np.tile(10, (25,)), RI_soc_m, '*')
            for n, H_test_factor in enumerate(hyper_H_values):
                plt.plot(np.tile(n+1, (25,)), RI_tablet_m_hyper[n], '*')
                plt.plot(np.tile(5+n+1, (25,)), RI_mech_m_hyper[n], '*')
                plt.plot(np.tile(10+n+1, (25,)), RI_soc_m_hyper[n], '*')
            plt.savefig(os.path.join(evaluation_folder, 'RI-change_hyper.png'))
            plt.close()
            """
            
            # RI_tablet_h ~ hyper
            # RI_mech_h   ~ 
            # RI_soc_h    ~ hypo    ~ default

            data = [RI_soc_h - RI_mech_h, RI_soc_h - RI_tablet_h]
            labels = ["Human:S-M", "Human:S-T"]
            for n, H_test in enumerate(hyper_H_values):
                data.append(RI_soc_m - RI_soc_m_hyper[n])
                labels.append(str(H_test))
            boxplot_fct(data, os.path.join(evaluation_folder, 'modify-prior_soc-hyper.pdf'), labels, title="Network with training prior reliance (H): " + str(H_train), figsize=(15,10))
            np.save(os.path.join(evaluation_folder, "modify-prior_soc-hyper.npy"), data)
            
            
            colors = sns.color_palette('husl', n_colors=len(hypo_H_values))
            labels = [str(i) for i in hypo_H_values]
            plt.figure()
            plt.plot([0, 1], [0, 1])
            for i in range(len(RI_tablet_m_hypo)):
                plt.plot(RI_tablet_m_hypo[i], RI_tablet_m, '*', color=colors[i], label=labels[i])
            plt.legend()
            plt.savefig(os.path.join(evaluation_folder, 'RI-direction-hypo.png'))
            plt.close()
            
            colors = sns.color_palette('husl', n_colors=len(hyper_H_values))
            labels = [str(i) for i in hyper_H_values]
            plt.figure()
            plt.plot([0, 1], [0, 1])
            for i in range(len(RI_tablet_m_hyper)):
                plt.plot(RI_tablet_m_hyper[i], RI_tablet_m, '*', color=colors[i], label=labels[i])
            plt.legend()
            plt.savefig(os.path.join(evaluation_folder, 'RI-direction-hyper.png'))
            plt.close()

        # write all results to csv files for statistics analyses
        with open("results/RI_difference_model_hyper-prior_net-" + str(folder_id) + ".csv", 'w') as f:
            f.write("subject\twho\tnet\tcomparison\tvalue\n")
#            f.write("alteredHprior\twho\tsubject\tnet\tdifference\n")
            for n in range(len(RI_soc_m_hyper)):
                diff = RI_soc_m - RI_soc_m_hyper[n]
                for i in range(len(diff)):
                    f.write(str(i) + "\tmodel\t" + str(folder_id) + "\t" + str(hyper_H_values[n]) + "\t" + str(diff[i]))
                    f.write('\n')


    # TEST CHANGE OF EXTERNAL SIGNAL VARIANCE INSTEAD OF PRIOR #
    ############################################################

        if modify_sigma_inp:
        # hyper-prior is a stronger prior, same as an increase of sigma
        #decrease_sigma_inp_factor = [0.1, 0.01, 0.001, 0.0001]
        #increase_sigma_inp_factor = [1/0.7, 1/0.3, 10] # == decrease of sensory precision through increase of sigma
            print('H_train: ' + str(H_train))
            network_to_load_folder = os.path.join(network_to_load, str(H_train))
            print(network_to_load_folder)
            evaluation_folder = os.path.join(network_to_load_folder, "evaluation")
            pathlib.Path(evaluation_folder).mkdir(parents=True, exist_ok=True)
            
            try:
                params, train_model = load_network(network_to_load_folder, model_filename='network-final.npz')
            except FileNotFoundError:
                print("Network file not found in: " + str(network_to_load_folder) + "\nContinue.")
                continue
                
            train_sigma_inp = train_model.external_signal_variance
            num_conditions = 3
            num_timesteps = 22
            num_participants = 25
            subjects = range(num_participants)

            initial_states_test = train_model.initial_states.W.array[classes_train,:]
            
            # regression indices of human and model
            
            try:
                if override:
                    raise FileNotFoundError('Force recreation of files')
                results = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_results.npy"))
                resultsv = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultsv.npy"))
                resultsm = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultsm.npy"))
                pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_pred_error.npy"))
                weighted_pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_weighted_pred_error.npy"))
                u_h_history = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_u_h_history.npy"))
                resultspos = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultspos.npy"))
                print('loaded existing')
            except FileNotFoundError:
                print('generate')
                results, resultsv, resultsm, pred_error, weighted_pred_error, u_h_history, resultspos = train_model.generate(initial_states_test, num_timesteps, external_input = x_train_norm, add_variance_to_output = add_variance_to_output, additional_output='activations', hyp_prior = H_train)
                
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_results.npy"), results)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultsv.npy"), resultsv)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultsm.npy"),  resultsm)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_pred_error.npy"), pred_error)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_weighted_pred_error.npy"), weighted_pred_error)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_u_h_history.npy"), u_h_history)
                np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_test-" + str(H_train) + "_resultspos.npy"), resultspos)


            presented_lengths = x_train_norm[:,-1] - x_train_norm[:,0]
            reproduced_lengths = results[:,-1] - results[:,0]
            human_reproduced_lengths = x_human_norm[:,-1] - x_human_norm[:,0]

            # calculate the regression indices from the model reproductions, and the human data
            model_RI, slope_model, intercept_model = calc_regression_indices(presented_lengths, reproduced_lengths, classes_train)
            human_RI, slope_human, intercept_human = calc_regression_indices(presented_lengths, human_reproduced_lengths, classes_train)
            
            # regression index for the different conditions
           
            RI_tablet_h = human_RI[0:num_participants*num_conditions:num_conditions]
            RI_mech_h = human_RI[1:num_participants*num_conditions:num_conditions]
            RI_soc_h = human_RI[2:num_participants*num_conditions:num_conditions]
            RI_tablet_m = model_RI[0:num_participants*num_conditions:num_conditions]
            RI_mech_m = model_RI[1:num_participants*num_conditions:num_conditions]
            RI_soc_m = model_RI[2:num_participants*num_conditions:num_conditions]

            human_diff_tablet_mechanical = RI_tablet_h - RI_mech_h
            human_diff_tablet_social = RI_tablet_h - RI_soc_h
            human_diff_mechanical_social = RI_mech_h - RI_soc_h
            model_diff_tablet_mechanical = RI_tablet_m - RI_mech_m
            model_diff_tablet_social = RI_tablet_m - RI_soc_m
            model_diff_mechanical_social = RI_mech_m - RI_soc_m

            boxplot_fct([model_diff_tablet_mechanical, human_diff_tablet_mechanical, model_diff_tablet_social, human_diff_tablet_social, model_diff_mechanical_social, human_diff_mechanical_social], os.path.join(evaluation_folder, 'model-human_comparison.pdf'), title="Network with training prior reliance (H): " + str(H_train))
            

            # Use initial states for tablet and replicate mechanical and social #
            #####################################################################
            # difference tablet IS, H / tablet IS, moderate hypo H == difference tablet / mechanical
            # difference tablet IS, H / tablet IS, strong hypo H == difference tablet / social

            decreaseSig_RIs = np.empty((len(decrease_sigma_inp),),dtype=object)
            RI_tablet_m_decreaseSig = np.empty((len(decrease_sigma_inp),),dtype=object)
            RI_mech_m_decreaseSig = np.empty((len(decrease_sigma_inp),),dtype=object)
            RI_soc_m_decreaseSig = np.empty((len(decrease_sigma_inp),),dtype=object)

            for n, sigma_inp in enumerate(decrease_sigma_inp):
                #sigma_inp = train_sigma_inp * sigma_inp_factor
                    
                print('sigma_inp: ' + "{:.1E}".format(sigma_inp))
                try:
                    if override:
                        raise FileNotFoundError('Force recreation of files')
                    results = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_results.npy"))
                    resultsv = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultsv.npy"))
                    resultsm = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultsm.npy"))
                    pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_pred_error.npy"))
                    weighted_pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_weighted_pred_error.npy"))
                    u_h_history = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_u_h_history.npy"))
                    resultspos = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultspos.npy"))
                    print('loaded existing')
                except FileNotFoundError:
                    print('File not found: ' + str(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_results.npy")))
                    results, resultsv, resultsm, pred_error, weighted_pred_error, u_h_history, resultspos = train_model.generate(initial_states_test, num_timesteps, external_input = x_train_norm, add_variance_to_output = add_variance_to_output, additional_output='activations', hyp_prior = H_train, external_signal_variance = sigma_inp)
                    
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_results.npy"), results)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultsv.npy"), resultsv)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultsm.npy"),  resultsm)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_pred_error.npy"), pred_error)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_weighted_pred_error.npy"), weighted_pred_error)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_u_h_history.npy"), u_h_history)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultspos.npy"), resultspos)
                
                reproduced_lengths = results[:,-1] - results[:,0]

                # calculate the regression indices from the model reproductions, and the human data
                model_RI, slope_model, intercept_model = calc_regression_indices(presented_lengths, reproduced_lengths, classes_train)
                decreaseSig_RIs[n] = model_RI
                RI_tablet_m_decreaseSig[n] = model_RI[0:num_participants*num_conditions:num_conditions]
                RI_mech_m_decreaseSig[n] = model_RI[1:num_participants*num_conditions:num_conditions]
                RI_soc_m_decreaseSig[n] = model_RI[2:num_participants*num_conditions:num_conditions]
                
                
                valid_trials, presented_lengths_valid_main, main_lengths, [u_h_history_split, classes_train_split, results_split, cond_list_split, subj_list_split] = identify_lengths(presented_lengths, [u_h_history, classes_train, results, cond_list, subj_list])

                # behavior
                #results_per_length_cond0, results_per_length_cond1, results_per_length_cond2 = sort_by_length_and_condition(results_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                #evaluate_activations_distances(results_per_length_cond0, results_per_length_cond1, results_per_length_cond2, evaluation_folder, "_behavior_sigma_sig-" + "{:.1E}".format(sigma_inp), train_model.num_c, plot_indiv = False, ymax=0.5)

                # activations
                full_data_per_length_cond0, full_data_per_length_cond1, full_data_per_length_cond2 = sort_by_length_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                evaluate_activations_distances(full_data_per_length_cond0, full_data_per_length_cond1, full_data_per_length_cond2, evaluation_folder, "_activations_H-" + str(H_train) + "_sig-" + "{:.1E}".format(sigma_inp), train_model.num_c, plot_indiv = False, ymax=0.5)
                #pca, scaler = evaluate_activations(u_h_history, initial_states_test, cond_list, subj_list, classes_train, evaluation_folder, "modify-sigma_sig-" + "{:.1E}".format(sigma_inp), num_timesteps, num_participants)

                # get data split by condition and by participant
                full_data_per_p_cond0, full_data_per_p_cond1, full_data_per_p_cond2 = sort_by_participant_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                evaluate_activations_distances(full_data_per_p_cond0, full_data_per_p_cond1, full_data_per_p_cond2, evaluation_folder, "_activations_H-" + str(H_train) + "_sig-" + "{:.1E}".format(sigma_inp) + "_per-participant", train_model.num_c, plot_indiv = False, ymax=0.5)


            # RI_tablet_h ~ hyper   ~ default
            # RI_mech_h   ~ 
            # RI_soc_h    ~ hypo
            
            data = [RI_tablet_h - RI_mech_h, RI_tablet_h - RI_soc_h]
            labels = ["Human:T-M", "Human:T-S"]
            for n, sigma_inp in enumerate(decrease_sigma_inp):
                data.append(RI_tablet_m - RI_tablet_m_decreaseSig[n])
                labels.append(str(sigma_inp))
            boxplot_fct(data, os.path.join(evaluation_folder, 'modify-sigma-inp_tablet-decreaseSig.pdf'), labels, xlabel="sigma external input", title="Network with training prior reliance (H): " + str(H_train), figsize=(15,10))
            np.save(os.path.join(evaluation_folder, "modify-sigma-inp_tablet-decreaseSig.npy"), data)


            # Use initial states for social and replicate tablet and mechanical #
            #####################################################################
            # difference social IS, H / social IS, moderate hyper H == difference social / mechanical
            # difference social IS, H / social IS, strong hyper H == difference social / tablet
            increaseSig_RIs = np.empty((len(increase_sigma_inp),),dtype=object)
            RI_tablet_m_increaseSig = np.empty((len(increase_sigma_inp),),dtype=object)
            RI_mech_m_increaseSig = np.empty((len(increase_sigma_inp),),dtype=object)
            RI_soc_m_increaseSig = np.empty((len(increase_sigma_inp),),dtype=object)

            for n, sigma_inp in enumerate(increase_sigma_inp):
                #sigma_inp = train_sigma_inp * sig_test_factor
                    
                print('sigma_inp: ' + "{:.1E}".format(sigma_inp))
                try:
                    results = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_results.npy"))
                    resultsv = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultsv.npy"))
                    resultsm = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultsm.npy"))
                    pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_pred_error.npy"))
                    weighted_pred_error = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_weighted_pred_error.npy"))
                    u_h_history = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_u_h_history.npy"))
                    resultspos = np.load(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultspos.npy"))
                    print('loaded existing')
                except FileNotFoundError:
                    print('generate')
                    results, resultsv, resultsm, pred_error, weighted_pred_error, u_h_history, resultspos = train_model.generate(initial_states_test, num_timesteps, external_input = x_train_norm, add_variance_to_output = add_variance_to_output, additional_output='activations', hyp_prior = H_train, external_signal_variance = sigma_inp)
                    
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_results.npy"), results)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultsv.npy"), resultsv)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultsm.npy"),  resultsm)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_pred_error.npy"), pred_error)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_weighted_pred_error.npy"), weighted_pred_error)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_u_h_history.npy"), u_h_history)
                    np.save(os.path.join(evaluation_folder, "evaluation_train-" + str(H_train) + "_sigma_inp-" + "{:.1E}".format(sigma_inp) + "_resultspos.npy"), resultspos)
                
                reproduced_lengths = results[:,-1] - results[:,0]

                # calculate the regression indices from the model reproductions, and the human data
                model_RI, slope_model, intercept_model = calc_regression_indices(presented_lengths, reproduced_lengths, classes_train)
                increaseSig_RIs[n] = model_RI
                RI_tablet_m_increaseSig[n] = model_RI[0:num_participants*num_conditions:num_conditions]
                RI_mech_m_increaseSig[n] = model_RI[1:num_participants*num_conditions:num_conditions]
                RI_soc_m_increaseSig[n] = model_RI[2:num_participants*num_conditions:num_conditions]
            
                valid_trials, presented_lengths_valid_main, main_lengths, [u_h_history_split, classes_train_split, results_split, cond_list_split, subj_list_split] = identify_lengths(presented_lengths, [u_h_history, classes_train, results, cond_list, subj_list])

                # behavior
                #results_per_length_cond0, results_per_length_cond1, results_per_length_cond2 = sort_by_length_and_condition(results_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                #evaluate_activations_distances(results_per_length_cond0, results_per_length_cond1, results_per_length_cond2, evaluation_folder, "_modify-sigma_sig-" + "{:.1E}".format(sigma_inp), train_model.num_c)

                # activations
                full_data_per_length_cond0, full_data_per_length_cond1, full_data_per_length_cond2 = sort_by_length_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                evaluate_activations_distances(full_data_per_length_cond0, full_data_per_length_cond1, full_data_per_length_cond2, evaluation_folder, "_activations_H-" + str(H_train) + "_sig-" + "{:.1E}".format(sigma_inp), train_model.num_c, plot_indiv = False, ymax=0.5)
                #pca, scaler = evaluate_activations(u_h_history, initial_states_test, cond_list, subj_list, classes_train, evaluation_folder, "modify-sigma_sig-" + "{:.1E}".format(sigma_inp), num_timesteps, num_participants)

                # get data split by condition and by participant
                full_data_per_p_cond0, full_data_per_p_cond1, full_data_per_p_cond2 = sort_by_participant_and_condition(u_h_history_split, classes_train_split, cond_list_split, subj_list_split, presented_lengths_valid_main, main_lengths, num_timesteps=22)
                evaluate_activations_distances(full_data_per_p_cond0, full_data_per_p_cond1, full_data_per_p_cond2, evaluation_folder, "_activations_H-" + str(H_train) + "_sig-" + "{:.1E}".format(sigma_inp) + "_per-participant", train_model.num_c, plot_indiv = False, ymax=0.5)

            # RI_tablet_h ~ hyper
            # RI_mech_h   ~ 
            # RI_soc_h    ~ hypo    ~ default

            data = [RI_soc_h - RI_mech_h, RI_soc_h - RI_tablet_h]
            labels = ["Human:S-M", "Human:S-T"]
            for n, sigma_inp in enumerate(increase_sigma_inp):
                data.append(RI_soc_m - RI_soc_m_increaseSig[n])
                labels.append(str(sigma_inp))
            boxplot_fct(data, os.path.join(evaluation_folder, 'modify-sigma-inp_soc-increaseSig.pdf'), labels, xlabel="sigma external input", title="Network with training prior reliance (H): " + str(H_train), figsize=(15,10))
            np.save(os.path.join(evaluation_folder, "modify-sigma-inp_soc-increaseSig.npy"), data)

        # write all results to csv files for statistics analyses
        with open("results/RI_difference_model_hyper-sensor_net-" + str(folder_id) + ".csv", 'w') as f:
            f.write("subject\twho\tnet\tcomparison\tvalue\n")
            for n in range(len(RI_soc_m_increaseSig)):
                diff = RI_soc_m - RI_soc_m_increaseSig[n]
                for i in range(len(diff)):
                    f.write(str(i) + "\tmodel\t" + str(folder_id) + "\t" + ("%.4f" % increase_sigma_inp[n]) + "\t" + str(diff[i]))
                    f.write('\n')

                
