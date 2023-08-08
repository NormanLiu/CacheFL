import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle
plt.rcParams.update({'font.size': 13})

plot_prefix = 'plots'
distribution = 'het'
#dataset_name = 'fmnist'
dataset_name = 'mnist.scale'
learning_rate = 0.04
# plotfile= plot_prefix + '/smallM-smallK-' + distribution + dataset_name + str(learning_rate) + 'long' + '-CNN.pdf'
plotfile1= plot_prefix + '/smallM-smallK-' + distribution + dataset_name + str(learning_rate) + 'sampling' + '-CNN-accuracy.pdf'
plotfile2= plot_prefix + '/smallM-smallK-' + distribution + dataset_name + str(learning_rate) + 'sampling' + '-CNN-loss.pdf'

plotfile1= plot_prefix + '/smallM-smallK-' + distribution + dataset_name + str(learning_rate) + 'large' + '-CNN-accuracy.pdf'
plotfile2= plot_prefix + '/smallM-smallK-' + distribution + dataset_name + str(learning_rate) + 'large' + '-CNN-loss.pdf'

M = 50

npzfile = np.load('random_delay_balance_less.npz')
# if dataset_name == 'mnist.scale':
#     #npzfile = np.load('random_delay_traces_asynch_mnist.npz')
#     npzfile = np.load('random_delay_balance_less.npz')
# elif dataset_name == 'cifar10':
#     npzfile = np.load('random_delay_traces_asynch_mnist.npz')
Kcache = npzfile['Kcache']
Kserver = npzfile['Kserver']
Tsingle_cacheFL = float(npzfile['Tsingle_cacheFL'])
Tsingle_less = float(npzfile['Tsingle_less'])
Tsingle_regular = float(npzfile['Tsingle_regular'])
Tsingle_random = npzfile['Tsingle_random']
with open("Kcache_list", "rb") as fp:
    Kcache_list = pickle.load(fp)
with open("Kserver_list", "rb") as fp:
    Kserver_list = pickle.load(fp)

loss_freq = 5
R = 200

Rs = [0] + list(range(loss_freq, R+1, loss_freq))

# generate delay for each case
TcacheFL = list(np.arange(0, (R + 1) * Tsingle_cacheFL, loss_freq * Tsingle_cacheFL))
Tregular = list(np.arange(0, (R + 1) * Tsingle_regular, loss_freq * Tsingle_regular))
Tless = list(np.arange(0, (R + 1) * Tsingle_less, loss_freq * Tsingle_less))
Trandom = [0]
current_T = 0
for r in range(R//loss_freq):
    current_T += loss_freq * Tsingle_random[r%20]
    Trandom.append(current_T)

alg_set = ['delayed', 'fedprox', 'random', 'straggler', 'fedavg']
l_set = ['delay_l', 'prox_l', 'random_l', 'less_l', 'local_l']
a_set = ['delay_a', 'prox_a', 'random_a', 'less_a', 'local_a']
loss_list = []
acc_list = []
for i in range(5):
    try:
        #savearray = plot_prefix + '/smallM-smallK-' + distribution + dataset_name + alg_set[i] + str(learning_rate) + 'sampling' + '-CCN.npz'
        savearray = plot_prefix + '/smallM-smallK-' + distribution + dataset_name + alg_set[i] + str(
            learning_rate) + 'large' + '-CCN.npz'
        npzfile = np.load(savearray)
        loss_list.append(npzfile[l_set[i]])
        acc_list.append(npzfile[a_set[i]])
    except:
        pass

# plot results
#plt.xscale("log")
fig = plt.figure(figsize=(12.8, 4.5))

ax = fig.add_subplot(121)
ax.plot(Rs[1:], acc_list[4], "v-", label='FedAvg')
ax.plot(Rs[1:], acc_list[0], "o-", label='CacheFL-OPT')
ax.plot(Rs[1:], acc_list[3], "s-", label='FedAvg stragglers')
ax.plot(Rs[1:], acc_list[1], "x-", label='FedProx')
ax.plot(Rs[1:], acc_list[2], "^-", label='CacheFL-random')
handles,labels_fig = ax.get_legend_handles_labels()
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Test Accuracy')
#ax.set_title('K={:d}, tau={:d}'.format(M, K))
ax.legend(handles, labels_fig, loc='lower right')
plt.xlim([Rs[8], Rs[-1]])

ax = fig.add_subplot(122)
ax.plot(Tregular[1:], acc_list[4], "v-", label='FedAvg')
ax.plot(TcacheFL[1:], acc_list[0], "o-", label='CacheFL-OPT')
ax.plot(Tless[1:], acc_list[3], "s-", label='FedAvg stragglers')
ax.plot(Tregular[1:], acc_list[1], "x-", label='FedProx')
ax.plot(Trandom[1:], acc_list[2], "^-", label='CacheFL-random')
handles, labels = ax.get_legend_handles_labels()
ax.set_xlabel('Wall-clock time')
ax.set_ylabel('Test Accuracy')
# ax.set_title('K={:d}, tau={:d}'.format(M, K))
ax.legend(handles, labels, loc='lower right')
plt.xlim([TcacheFL[8], TcacheFL[-1]])

plt.savefig(plotfile1, format="pdf", bbox_inches="tight")



fig = plt.figure(figsize=(12.8, 4.5))

ax = fig.add_subplot(121)
ax.plot(Rs[1:], loss_list[4], "v-", label='FedAvg')
ax.plot(Rs[1:], loss_list[0], "o-", label='CacheFL-OPT')
ax.plot(Rs[1:], loss_list[3], "s-", label='FedAvg stragglers')
ax.plot(Rs[1:], loss_list[1], "x-", label='FedProx')
ax.plot(Rs[1:], loss_list[2], "^-", label='CacheFL-random')
handles,labels_fig = ax.get_legend_handles_labels()
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Loss Value')
#ax.set_title('K={:d}, tau={:d}'.format(M, K))
ax.legend(handles, labels_fig, loc='lower right')
plt.xlim([Rs[8], Rs[-1]])

ax = fig.add_subplot(122)
ax.plot(Tregular[1:], loss_list[4], "v-", label='FedAvg')
ax.plot(TcacheFL[1:], loss_list[0], "o-", label='CacheFL-OPT')
ax.plot(Tless[1:], loss_list[3], "s-", label='FedAvg stragglers')
ax.plot(Tregular[1:], loss_list[1], "x-", label='FedProx')
ax.plot(Trandom[1:], loss_list[2], "^-", label='CacheFL-random')
handles, labels = ax.get_legend_handles_labels()
ax.set_xlabel('Wall-clock time')
ax.set_ylabel('Loss Value')
# ax.set_title('K={:d}, tau={:d}'.format(M, K))
ax.legend(handles, labels, loc='lower right')
plt.xlim([TcacheFL[8], TcacheFL[-1]])

plt.savefig(plotfile2, format="pdf", bbox_inches="tight")