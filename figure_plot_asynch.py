import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle
plt.rcParams.update({'font.size': 14})

plot_prefix = 'plots'
distribution = 'het'
dataset_name = 'mnist.scale'
agg_scheme = 2
plotfile= plot_prefix + '/smallM-smallK-asynch-' + distribution + dataset_name + str(agg_scheme) + '.pdf'
savearray=plot_prefix + '/smallM-smallK-asynch-' + distribution + dataset_name + str(agg_scheme) + '.npz'


M = 50

if dataset_name == 'mnist.scale':
    # npzfile = np.load('random_delay_traces_asynch_mnist.npz')
    npzfile = np.load('random_delay_balance_less.npz')
elif dataset_name == 'cifar10':
    npzfile = np.load('random_delay_traces_asynch_cifar10.npz')
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
Tsingle_random = np.concatenate((Tsingle_random, Tsingle_random))
Kcache_list.extend(Kcache_list)
Kserver_list.extend(Kserver_list)
Tsingle_k = npzfile['Tsingle_k']
timestamp_list = []
clientAction_list = []
next_update = Tsingle_k.copy()
for _ in range(M * 200 * 2):
    next_time = min(next_update)
    next_k = np.argmin(next_update)
    timestamp_list.append(next_time)
    clientAction_list.append(next_k)
    next_update[next_k] += Tsingle_k[next_k]

loss_freq = 2
R = 60

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
Tasynch = [0] + [timestamp_list[i*M*loss_freq] for i in range((R * 2)//loss_freq)]

if TcacheFL[-1] < Trandom[-1]:
    Tregular = [x for x in Tregular if x <= TcacheFL[-1]]
    Trandom = [x for x in Trandom if x <= TcacheFL[-1]]
    Tasynch = [x for x in Tasynch if x <= TcacheFL[-1]]

npzfile = np.load(savearray)
local_l = npzfile['local_l']
delay_l = npzfile['delay_l']
less_l = npzfile['less_l']
prox_l = npzfile['prox_l']
random_l = npzfile['random_l']
asynch_l = npzfile['asynch_l']

local_a = npzfile['local_a']
delay_a = npzfile['delay_a']
less_a = npzfile['less_a']
prox_a = npzfile['prox_a']
random_a = npzfile['random_a']
asynch_a = npzfile['asynch_a']

# plot results
plt.xscale("log")
fig = plt.figure(figsize=(12.8, 4.5))

ax = fig.add_subplot(121)
ax.semilogx(Tregular, local_l[0:len(Tregular)], "v-", label='FedAvg')
ax.semilogx(TcacheFL, delay_l[0:len(TcacheFL)], "o-", label='CacheFL-OPT')
ax.semilogx(Tless, less_l[0:len(Tless)], "s-", label='FedAvg stragglers')
ax.semilogx(Tregular, prox_l[0:len(Tregular)], "x-", label='FedProx')
ax.semilogx(Trandom, random_l[0:len(Trandom)], "^-", label='CacheFL-random')
ax.semilogx(Tasynch, asynch_l[0:len(Tasynch)], "--", label='FedAsync')
handles, labels = ax.get_legend_handles_labels()
ax.set_xlabel('Wall-clock Time')
ax.set_ylabel('Training Loss')
#ax.set_title('K={:d}, tau={:d}'.format(M, K))
ax.legend(handles, labels, loc='upper right')
plt.xlim([Tregular[1], Tregular[-1]])

ax = fig.add_subplot(122)
ax.semilogx(Tregular, local_a[0:len(Tregular)], "v-", label='FedAvg')
ax.semilogx(TcacheFL, delay_a[0:len(TcacheFL)], "o-", label='CacheFL-OPT')
ax.semilogx(Tless, less_a[0:len(Tless)], "s-", label='FedAvg stragglers')
ax.semilogx(Tregular, prox_a[0:len(Tregular)], "x-", label='FedProx')
ax.semilogx(Trandom, random_a[0:len(Trandom)], "^-", label='CacheFL-random')
ax.semilogx(Tasynch, asynch_a[0:len(Tasynch)], "--", label='FedAsync')
handles, labels = ax.get_legend_handles_labels()
ax.set_xlabel('Wall-clock time')
ax.set_ylabel('Test Accuracy')
# ax.set_title('K={:d}, tau={:d}'.format(M, K))
ax.legend(handles, labels, loc='lower right')

plt.xlim([Tregular[1], Tregular[-1]])
plt.ylim(0.5, 0.9)

plt.savefig(plotfile, format="pdf", bbox_inches="tight")