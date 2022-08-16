import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy

def plot_with_covariance(data, cond_list, filename):

    fig = plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20, 'xtick.labelsize': 15, 'ytick.labelsize': 15})
    ax = fig.add_subplot(111)
    #colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'darkorange', 'navy', 'yellow', 'black']
    colors = ['r', '#f3a712', 'b']
    markers = ['^', 's', 'd']
    lw = 2

    # plot scatter dots
    #for i,is_id in enumerate(classes_train):
    #    current_cond = cond_list[i]
    #    plt.scatter(data[i,0], data[i,1], marker='*', color=colors[current_cond])

    data_split_0 = data[np.where(cond_list==0)[0],:]
    data_split_1 = data[np.where(cond_list==1)[0],:]
    data_split_2 = data[np.where(cond_list==2)[0],:]

    plt.scatter(data_split_1[:,0], data_split_1[:,1], marker=markers[1], color=colors[1], s=10)
    plt.scatter(data_split_2[:,0], data_split_2[:,1], marker=markers[2], color=colors[2], s=10)
    plt.scatter(data_split_0[:,0], data_split_0[:,1], marker=markers[0], color=colors[0], s=10)
    
    mean_0 = np.mean(data_split_0, axis=0)
    cov_0 = np.cov(np.transpose(data_split_0))
    mean_1 = np.mean(data_split_1, axis=0)
    cov_1 = np.cov(np.transpose(data_split_1))
    mean_2 = np.mean(data_split_2, axis=0)
    cov_2 = np.cov(np.transpose(data_split_2))

    plt.scatter(mean_0[0], mean_0[1], marker='*', color='k', s=50)
    plt.scatter(mean_1[0], mean_1[1], marker='v', color='k', s=50)
    plt.scatter(mean_2[0], mean_2[1], marker='s', color='k', s=50)


    v,w = scipy.linalg.eigh(cov_0)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / scipy.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi # convert to degrees
    ell = mpl.patches.Ellipse(mean_0, v[0], v[1], 180. + angle, edgecolor = 'black', facecolor='none')
    #ell.set_clip_box(plt.bbox)
    ell.set_alpha(1)
    plt.gca().add_patch(ell)


    v,w = scipy.linalg.eigh(cov_1)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / scipy.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi # convert to degrees
    ell = mpl.patches.Ellipse(mean_1, v[0], v[1], 180. + angle, edgecolor = 'black', facecolor='none')
    #ell.set_clip_box(plt.bbox)
    ell.set_alpha(1)
    plt.gca().add_patch(ell)
    
    
    v,w = scipy.linalg.eigh(cov_2)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / scipy.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi # convert to degrees
    ell = mpl.patches.Ellipse(mean_2, v[0], v[1], 180. + angle, edgecolor = 'black', facecolor='none')
    #ell.set_clip_box(plt.bbox)
    ell.set_alpha(1)
    plt.gca().add_patch(ell)

    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

