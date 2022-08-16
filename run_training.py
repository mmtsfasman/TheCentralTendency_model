# General imports
from distutils.file_util import copy_file
import itertools
import numpy as np
import os
import pathlib
import sys
import time
import datetime
import random
import seaborn as sns 

#Matplotlib imports and default layout definition
import matplotlib
matplotlib.rcParams['figure.dpi'] = 75
COLOR = 'grey'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['ytick.left'] = True
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['axes.edgecolor'] = '#dddddd'
matplotlib.rcParams['ytick.color'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['figure.facecolor'] = 'white'
# to enable plotting on a remote machine, without a running X connection:
if not 'matplotlib.pyplot' in sys.modules:
   matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Imports for neural network training
import chainer
from chainer.backends import cuda
from chainer import optimizers, serializers
from chainer.functions.math import exponential
from nets import SCTRNN, make_initial_state_zero, make_initial_state_random, NetworkParameterSetting, save_network, load_network
from utils.distance_measures import distance_measure
from utils.normalize import normalize, range2norm, norm2range
from utils.visualization import plot_results, plot_pca_activations
#from get_central_tendency_data import get_central_tendency_data

# Determine whether CPU or GPU should be used
gpu_id = 0 # -1 for cpu
xp = np
if gpu_id >= 0 and cuda.available:
   print("Use GPU!")
   #If NVRTCError:
   #$ export CUDA_HOME=/usr/local/cuda-9.1
   #$ export PATH=${CUDA_HOME}/bin:${PATH}
   #$ export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
   cuda.get_device_from_id(gpu_id).use()
   xp = cuda.cupy
else:
   print("Use CPU!")
   gpu_id = -1

data_set_name = "all_human"

# whether to add the explicit_sensor_variance to the training signal
add_external_signal_variance = False

# Hyper-prior parameter that influences the Bayesian inference of the network
hyp_prior_runs = [1]
runs = len(hyp_prior_runs)
reuse_existing_weights = False

# weight and bias are adapted during learning
learn_bias = True

# How the prediction error is computed:
#  'standard': compare with ground truth
#  'integrated': compare with posterior of perception of ground truth
#  'stimulus-human': compare with human behavior
prediction_error_type = 'stimulus-human'

save_interval = 100   # interval for testing the production capability of the network and saving initial state information and model
epochs = 15000        # total maximum number of epochs
num_timesteps = 22    # time steps for each reproduction
num_io = 1            # input/output dimension
num_participants = 25 # number of participants
subjects = range(num_participants)

# load the presented data (input) and the human behavior (learning target)
x_train_norm = np.float32(np.load('./human_data/presented_norm.npy'))
x_human_norm = np.float32(np.load('./human_data/human_norm.npy'))
# mapping all data entries to the initial state that should be used
classes_train = np.load('./human_data/classes.npy')
# mapping all data entries to conditions: 0 (individual), 1 (mechanical), 2 (social)
cond_list = np.load('./human_data/conditions.npy')
# mapping all data entries to subjects
subj_list = np.load('./human_data/subjects.npy')

num_classes = xp.int(np.max(classes_train)+1)
num_all_samples = len(classes_train)

save_location = os.path.join("./results/training", data_set_name)
now = datetime.datetime.now()
expStr = "human_training_" + '_'.join(map(str, subjects)) + "_" + str(now.year).zfill(4) + "-" + str(now.month).zfill(2) + "-" + str(now.day).zfill(2) + "_" + str(now.hour).zfill(2) + "-" + str(now.minute).zfill(2) + "_" + str(now.microsecond).zfill(7)
save_dir = os.path.join(save_location, expStr)

for r in range(runs):
    best_epoch_error = np.Infinity # some high initial error value
    best_epoch = 0

    # Explicit sensor variance for BI
    explicit_sensor_variance = 0.001 #0.001

    # Hypo prior: variance added for BI
    if len(hyp_prior_runs) > 0:
        hyp_prior = hyp_prior_runs[r]

    final_save_dir = os.path.join(save_dir, str(hyp_prior))
    pathlib.Path(final_save_dir).mkdir(parents=True, exist_ok=True)

    x_train = np.copy(x_train_norm)
    x_human = np.copy(x_human_norm)
    
    # adding sensor variance and set the sensor variance accordingly to mimic "accurate perception"
    external_signal_variance_vec = xp.ones((x_train.shape)) * explicit_sensor_variance
    print("Update explicit sensor variance to actual sensor variance " + str(explicit_sensor_variance))

    if add_external_signal_variance:
        if gpu_id >= 0:
            x_train = cuda.to_gpu(x_train) + xp.sqrt(external_signal_variance_vec) * xp.random.randn(x_train.shape[0], x_train.shape[1])
            x_train = xp.float32(cuda.to_cpu(x_train))
        else:
            x_train += xp.sqrt(external_signal_variance_vec) * xp.random.randn(x_train.shape[0], x_train.shape[1])

    c_train = classes_train

    # how many samples to present in each batch: all
    batch_size = len(x_train)

    # Initialize parameter setting
    training_ext_contrib = 1
    training_tau = 2
    training_context_n = 25 # number of context layer neurons
    init_state_var = 1e7
    aberrant_sensory_precision = 0
    excitation_bias = 1/training_context_n # default 0.05
    lrate = 0.001
    conn = training_context_n
    var_integration = 2

    # CREATE PARAMETER SETTING AND NETWORK MODEL
    p = NetworkParameterSetting(epochs = epochs, batch_size = batch_size)
    p.set_network_type('SCTRNN', {'num_io':x_train.shape[1]/num_timesteps, 'num_c':training_context_n, 'lr':lrate, 'num_classes': num_classes,
       'learn_tau':False, 'tau_c':training_tau,
       'learn_init_states':True, 'init_state_init':'zero', 'init_state_var': init_state_var,
       'learn_weights':True,
       'learn_bias':learn_bias,
       'hyp_prior':hyp_prior,
       'external_signal_variance':explicit_sensor_variance})

    with open(os.path.join(final_save_dir,"info.txt"),'w') as f:
       f.write(p.get_parameter_string())
    f.close()

    # create new RNN model
    model = SCTRNN(p.num_io, p.num_c, p.tau_c, p.num_classes, init_state_init = p.init_state_init, init_state_learning = p.learn_init_states, weights_learning = p.learn_weights, bias_learning = p.learn_bias, tau_learning = p.learn_tau, hyp_prior = p.hyp_prior, external_signal_variance = p.external_signal_variance)
    model.add_BI_variance = True
    model.set_init_state_learning(c_train)

    # store weights of first run, to be reused by next runs
    if runs > 1 and same_weights_per_run or reuse_existing_weights:
        if r == 0 and not reuse_existing_weights:
            xhW=model.x_to_h.W.data
            hhW=model.h_to_h.W.data
            hyW=model.h_to_y.W.data
            hvW=model.h_to_v.W.data
            np.save(os.path.join(init_weight_dir, 'xhW.npy'), xhW)
            np.save(os.path.join(init_weight_dir, 'hhW.npy'), hhW)
            np.save(os.path.join(init_weight_dir, 'hyW.npy'), hyW)
            np.save(os.path.join(init_weight_dir, 'hvW.npy'), hvW)
        else:
            print("Load predefined initial weights from " + init_weight_dir)
            xhW=np.load(os.path.join(init_weight_dir, 'xhW.npy'))
            hhW=np.load(os.path.join(init_weight_dir, 'hhW.npy'))
            hyW=np.load(os.path.join(init_weight_dir, 'hyW.npy'))
            hvW=np.load(os.path.join(init_weight_dir, 'hvW.npy'))
            model.x_to_h.W.data=xhW[:model.num_c, :]
            model.h_to_h.W.data=hhW[:model.num_c, :model.num_c]
            model.h_to_y.W.data=hyW[:, :model.num_c]
            model.h_to_v.W.data=hvW[:, :model.num_c]

    if runs > 1 and same_bias_per_run or reuse_existing_weights:
        if r == 0 and not reuse_existing_weights:
            xhb=model.x_to_h.b.data
            hhb=model.h_to_h.b.data
            hyb=model.h_to_y.b.data
            hvb=model.h_to_v.b.data
            np.save(os.path.join(init_weight_dir, 'xhb.npy'), xhb)
            np.save(os.path.join(init_weight_dir, 'hhb.npy'), hhb)
            np.save(os.path.join(init_weight_dir, 'hyb.npy'), hyb)
            np.save(os.path.join(init_weight_dir, 'hvb.npy'), hvb)
        else:
            print("Load predefined initial bias weights from " + init_weight_dir)
            xhb=np.load(os.path.join(init_weight_dir, 'xhb.npy'))
            hhb=np.load(os.path.join(init_weight_dir, 'hhb.npy'))
            hyb=np.load(os.path.join(init_weight_dir, 'hyb.npy'))
            hvb=np.load(os.path.join(init_weight_dir, 'hvb.npy'))
            model.x_to_h.b.data=xhb[:model.num_c]
            model.h_to_h.b.data=hhb[:model.num_c]
            model.h_to_y.b.data=hyb
            model.h_to_v.b.data=hvb

    save_network(final_save_dir, params=p, model = model, model_filename = "network-initial")

    if gpu_id >= 0:
       model.to_gpu(gpu_id)

    # Optimizer: takes care of updating the model using backpropagation
    optimizer = optimizers.Adam(p.lr) # optimizers.RMSprop()
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0))

    history_init_state_var = np.zeros((epochs+1,))
    history_init_state_var[0] = np.mean(np.var(model.initial_states.W.array,axis=0))
    history_generation_error_proactive = np.empty((p.num_classes,), dtype=object)
    history_generation_error_reactive = np.empty((p.num_classes,), dtype=object)
    history_training_error = np.zeros((epochs+1,))
    history_training_variance_estimation = np.zeros((epochs+1,))

    print("actual variance of init_states_0: " + str(history_init_state_var[0]))

    # create subset of indices to evaluate efficiently current learning status
    num_eval_samples = 200
    eval_index_set = np.unique([int(np.floor(np.random.rand()*x_human.shape[0])) for i in range(num_eval_samples)])

    # Evaluate the performance of the untrained network
    res, resv, resm, pe, wpe, u_h_history, respos = model.generate(model.initial_states.W.array[classes_train[eval_index_set]], num_timesteps, external_input = xp.copy(xp.asarray(x_train))[eval_index_set], add_variance_to_output = 0, additional_output='activations', external_signal_variance = explicit_sensor_variance)
    res = cuda.to_cpu(res)
    dist_train = x_train[eval_index_set][:,-1] - x_train[eval_index_set][:,0]
    dist_human = x_human[eval_index_set][:,-1] - x_human[eval_index_set][:,0]
    dist_res = res[:,-1] - res[:,0]
    #print(dist_human)
    diff_res_to_human = np.mean(np.abs(dist_human - dist_res))
    diff_res_to_train = np.mean(np.abs(dist_train - dist_res))
    diff_human_to_train = np.mean(np.abs(dist_train - dist_human))
    with open(os.path.join(final_save_dir, "evaluation-human.txt"), 'a') as f:
      f.write("Epoch 1:\nres-to-human " + str(diff_res_to_human) + "\nres-to-train " + str(diff_res_to_train) + "\ndiff_human_to_train " + str(diff_human_to_train) + "\n")
    
    # error between human data and network output
    error_history = []
    
    for epoch in range(1, p.epochs + 1):
        epoch_start_time = time.time()

        outv = np.zeros((num_timesteps,))

        x_train = np.copy(x_train_norm)
        x_human = np.copy(x_human_norm)

        if add_external_signal_variance:
            if gpu_id >= 0:
                x_train = cuda.to_gpu(x_train) + xp.sqrt(external_signal_variance_vec) * xp.random.randn(x_train.shape[0], x_train.shape[1])
                x_train = xp.float32(cuda.to_cpu(x_train))
            else:
                x_train += xp.sqrt(external_signal_variance_vec) * xp.random.randn(x_train.shape[0], x_train.shape[1])
                
        #print(np.max([np.mean(np.var(x_train[x:num_all_samples:num_classes,:],axis=0)) for x in range(num_classes)]))

        # permutate samples in each epoch so that they are randomly ordered
        perm = np.random.permutation(x_train.shape[0])

        for b in range(0, len(x_train), p.batch_size):
            batch_start_time = time.time()
            b_end = np.min([b+p.batch_size, len(x_train)])
            
            x_batch = xp.asarray(x_train[perm][b:b_end,:])
            x_batch_human = xp.asarray(x_human[perm][b:b_end,:])
            
            # tell the model which index of the training data will be for which class
            model.set_init_state_learning(c_train[perm][b:b_end])

            mean_init_states = chainer.Variable(xp.zeros((),dtype=xp.float32))
            mean_init_states = chainer.functions.average(model.initial_states.W,axis=0)

            # initialize loss
            acc_loss = chainer.Variable(xp.zeros((),dtype=xp.float32)) # for weight backprop
            acc_init_loss = chainer.Variable(xp.zeros((),dtype=xp.float32)) # for init states backprop
            acc_bias_loss = chainer.Variable(xp.zeros((),dtype=xp.float32)) # for keeping the bias distribution at a desired variance
            acc_conn_loss = chainer.Variable(xp.zeros((),dtype=xp.float32)) # for keeping the network weights at a certain connectivity
            err = xp.zeros(()) # for evaluation only

            # clear gradients from previous batch
            model.cleargrads()
            # clear output and variance estimations from previous batch
            model.reset_current_output()

            t=0 # iterate through time
            x_t = x_batch[:, p.num_io*t:p.num_io*(t+1)]
            # next time step to be predicted (for evaluation)
            x_t1 = x_batch[:, p.num_io*(t+1):p.num_io*(t+2)]
            
            x_t_human = x_batch_human[:, p.num_io*t:p.num_io*(t+1)]
            # next time step to be predicted (for evaluation)
            x_t1_human = x_batch_human[:, p.num_io*(t+1):p.num_io*(t+2)]

            # execute first forward step
            u_h, y, v = model(xp.copy(x_t), None) # initial states of u_h are set automatically according to model.classes

            # noisy output estimation
            # y_out = y.array + xp.sqrt(v.array) * xp.random.randn()

            # compute prediction error, averaged over batch
            if prediction_error_type == 'stimulus-human':
                # compare output to ground truth of human
                loss_i = chainer.functions.gaussian_nll(chainer.Variable(x_t1_human), y, exponential.log(v))
            elif prediction_error_type == 'standard':
                # compare output to ground truth
                loss_i = chainer.functions.gaussian_nll(chainer.Variable(x_t1), y, exponential.log(v))
            elif prediction_error_type == 'integrated':
                # compare output to posterior of perception
               if var_integration == 1:
                   integrated_x = p.training_external_contrib * chainer.Variable(x_t1) + (1 - p.training_external_contrib) * (y + chainer.functions.sqrt(v) * xp.random.randn())
                   loss_i = chainer.functions.gaussian_nll(integrated_x, y, exponential.log(v))
               elif var_integration == 2:
                   loss_i = chainer.functions.gaussian_nll(model.current_x, y, exponential.log(v))

            acc_loss += loss_i

            # compute error for evaluation purposes
            err += chainer.functions.mean_squared_error(chainer.Variable(x_t1_human), y).array.reshape(()) * p.batch_size

            outv[t] = xp.mean(v.array)

            # rollout trajectory
            for t in range(1,num_timesteps-1):
               # current time step
               x_t = x_batch[:, p.num_io*t:p.num_io*(t+1)]
               # next time step to be predicted (for evaluation)
               x_t1 = x_batch[:, p.num_io*(t+1):p.num_io*(t+2)]
               
               # current time step
               x_t_human = x_batch_human[:, p.num_io*t:p.num_io*(t+1)]
               x_t1_human = x_batch_human[:, p.num_io*(t+1):p.num_io*(t+2)]

               u_h, y, v = model(xp.copy(x_t), u_h)

               # noisy output estimation
               # y_out = y.array + xp.sqrt(v.array) * xp.random.randn()
                
               if prediction_error_type == 'stimulus-human':
                   loss_i = chainer.functions.gaussian_nll(chainer.Variable(x_t1_human), y, exponential.log(v))
               # compute error for backprop for weights
               elif prediction_error_type == 'standard':
                   loss_i = chainer.functions.gaussian_nll(chainer.Variable(x_t1), y, exponential.log(v))
               elif prediction_error_type == 'integrated':
                    if var_integration == 1:
                        integrated_x = p.training_external_contrib * chainer.Variable(x_t1) + (1 - p.training_external_contrib) * (y + chainer.functions.sqrt(v) * xp.random.randn())
                        loss_i = chainer.functions.gaussian_nll(integrated_x, y, exponential.log(v))
                    elif var_integration == 2:
                        loss_i = chainer.functions.gaussian_nll(model.current_x, y, exponential.log(v))
               acc_loss += loss_i

               # compute error for evaluation purposes
               err += chainer.functions.mean_squared_error(chainer.Variable(x_t_human), y).array.reshape(()) * p.batch_size

               outv[t] = xp.mean(v.array)

            # for each training sequence of this batch: compute loss for maintaining desired initial state variance
            for s in range(model.initial_states().shape[0]):
               if gpu_id >= 0:
                   acc_init_loss += chainer.functions.gaussian_nll(model.initial_states()[model.classes][s], mean_init_states, exponential.log(cuda.to_gpu(p.init_state_var, device=gpu_id)))
               else:
                   acc_init_loss += chainer.functions.gaussian_nll(model.initial_states()[model.classes][s], mean_init_states, exponential.log(p.init_state_var))

            # compute gradients
            # (gradients from L_out and L_init are summed up)
            # gradient of initial states equals:
            # 1/p.init_state_var * (c0[cl]-mean_init_states).array

            acc_init_loss.backward()
            acc_loss.backward()
            optimizer.update()
            
            print("Elapsed time (mini-batch): " + str(time.time() - batch_start_time))

        # all batches of this epoch finished
        print("Done epoch " + str(epoch))
        print("Elapsed time (full epoch): " + str(time.time() - epoch_start_time))
        error = err/p.batch_size/num_timesteps
        mean_estimated_var = xp.mean(outv)
        history_training_error[epoch] = error
        history_training_variance_estimation[epoch] = mean_estimated_var

        print("train MSE = " + str(error) + "\nmean estimated var: " + str(mean_estimated_var))
        #print("init_states = [" + str(model.initial_states.W.array[0][0]) + "," + str(model.initial_states.W.array[0][1]) + "...], var: " + str(np.mean(np.var(model.initial_states.W.array,axis=0))) + ", accs: " + str(acc_loss) + " + " + str(acc_init_loss) + " + " + str(acc_bias_loss))

        history_init_state_var[epoch] = xp.mean(np.var(model.initial_states.W.array,axis=0))

        with open(os.path.join(final_save_dir,"evaluation.txt"),'a') as f:
           f.write("epoch: " + str(epoch)+ "\n")
           f.write("train MSE = " + str(error) + "\nmean estimated var: " + str(mean_estimated_var))
           f.write("initial state var: " + str(history_init_state_var[epoch]) + ", precision loss: " + str(acc_loss) + ", variance loss: " + str(acc_init_loss) + " + " + str(acc_bias_loss) + "\ninit states:\n")
           for i in range(p.num_classes):
               f.write("\t[" + str(model.initial_states.W[i][0]) + "," + str(model.initial_states.W[i][1]) + "...]\n")
        f.close()

        if epoch%save_interval == 1 or epoch == p.epochs:
            res, resv, resm, pe, wpe, u_h_history, respos = model.generate(model.initial_states.W.array[classes_train[eval_index_set]], num_timesteps, external_input = xp.copy(xp.asarray(x_train))[eval_index_set], add_variance_to_output = 0, additional_output='activations', external_signal_variance = explicit_sensor_variance)
            res = cuda.to_cpu(res)
            dist_train = x_train[eval_index_set][:,-1] - x_train[eval_index_set][:,0]
            dist_human = x_human[eval_index_set][:,-1] - x_human[eval_index_set][:,0]
            dist_res = res[:,-1] - res[:,0]
            #print(dist_human)
            diff_res_to_human = np.mean(np.abs(dist_human - dist_res))
            diff_res_to_train = np.mean(np.abs(dist_train - dist_res))
            diff_human_to_train = np.mean(np.abs(dist_train - dist_human))
            with open(os.path.join(final_save_dir, "evaluation-human.txt"), 'a') as f:
              f.write("Epoch " + str(epoch) + ":\nres-to-human " + str(diff_res_to_human) + "\nres-to-train " + str(diff_res_to_train) + "\ndiff_human_to_train " + str(diff_human_to_train) + "\n")

            error_history.append([epoch, diff_res_to_human])
            np.save(os.path.join(final_save_dir, "error_history.npy"), error_history)            
            
            plt.figure()
            plt.plot(np.asarray(error_history)[:,0], np.asarray(error_history)[:,1])
            plt.savefig(os.path.join(final_save_dir, 'error_history.png'))
            plt.close()


            #print(len(dist_human)) 
            #print(len(dist_res))  
            #import pdb; pdb.set_trace()
            marker = itertools.cycle(("v", "o", "s"))
            marker1 = itertools.cycle(("v", "o", "s"))
            colors = np.repeat(sns.color_palette('husl', n_colors=len(subjects)),3,axis=0)
           

            marker_list = ['v','o','s']
            colors = sns.color_palette('husl', n_colors=3)

            fig =  plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.arange(0,2,0.1), np.arange(0,2,0.1), color='c', alpha=0.4)
            for cl in range(num_classes):
              #print((classes_train == cl))
              ax.scatter(xp.reshape(dist_human, (len(dist_human), 1))[(classes_train[eval_index_set] == cl)], xp.reshape(dist_res, (len(dist_res), 1))[(classes_train[eval_index_set] == cl)], color=colors[cl%3], marker=marker_list[cl%3])
            ax.set_xlim([0, 2])
            ax.set_ylim([0, 2])
            plt.savefig(os.path.join(final_save_dir, "dist_human-to-res_epoch-" + str(epoch) + '.png'))
            plt.close()              

            fig =  plt.figure()
            ax = fig.add_subplot(111)
            for cl in range(num_classes):
              ax.scatter(xp.reshape(dist_train, (len(dist_train), 1))[(classes_train[eval_index_set] == cl)], xp.reshape(dist_res, (len(dist_res), 1))[(classes_train[eval_index_set] == cl)], color=colors[cl%3], marker=marker_list[cl%3])
            ax.plot(np.arange(0,2,0.1), np.arange(0,2,0.1), color='c', alpha=0.4)
            ax.set_xlim([0, 2])
            ax.set_ylim([0, 2])
            plt.savefig(os.path.join(final_save_dir, "dist_train-to-res_epoch-" + str(epoch) + '.png'))
            plt.close()          

            save_network(final_save_dir, p, model, model_filename="network-epoch-"+str(epoch).zfill(len(str(epochs))))
            np.save(os.path.join(final_save_dir,"history_init_state_var"), np.array(history_init_state_var))
            np.save(os.path.join(final_save_dir,"history_training_error"), np.array(history_training_error))
            np.save(os.path.join(final_save_dir, "history_training_variance_estimation"), np.array(history_training_variance_estimation[0:epoch]))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.arange(0,len(history_init_state_var)), history_init_state_var)
            plt.title("init state variance")
            fig.savefig(os.path.join(final_save_dir,"init-state-var"))
            plt.close()

    save_network(final_save_dir, p, model, model_filename = "network-final")
    plt.close('all')

