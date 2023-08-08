import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle
plt.rcParams.update({'font.size': 17})

plot_prefix = 'plots'
distribution = 'het'
dataset_name = 'synthetic'
plotfile= plot_prefix + '/smallM-smallK-' + distribution + dataset_name + '_traces' + '.png'
savearray=plot_prefix + '/smallM-smallK-' + distribution + dataset_name + '.npz'

#npzfile = np.load('random_delay.npz')
npzfile = np.load('random_delay_traces.npz') #with real communication traces
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
TcacheFL = list(np.arange(0, (R + 1) * Tsingle_cacheFL, loss_freq * Tsingle_cacheFL))
Tregular = list(np.arange(0, (R + 1) * Tsingle_regular, loss_freq * Tsingle_regular))
Tless = list(np.arange(0, (R + 1) * Tsingle_less, loss_freq * Tsingle_less))
Trandom = [0]
current_T = 0
for r in range(20):
    current_T += loss_freq * Tsingle_random[r]
    Trandom.append(current_T)
    current_T += loss_freq * Tsingle_random[r]
    Trandom.append(current_T)

if TcacheFL[-1] < Trandom[-1]:
    Tregular = [x for x in Tregular if x <= TcacheFL[-1]]
    Trandom = [x for x in Trandom if x <= TcacheFL[-1]]

Rs = [0] + list(range(loss_freq, R+1, loss_freq))

npzfile = np.load(savearray)
local_l = npzfile['local_l']
delay_l = npzfile['delay_l']
less_l = npzfile['less_l']
prox_l = npzfile['prox_l']
random_l = npzfile['random_l']

# plot results
fig = plt.figure(figsize=(6.4, 9.6))
ax = fig.add_subplot(211)
ax.plot(Rs, local_l, "v-", label='FedAvg')
ax.plot(Rs, delay_l, "o-", label='CacheFL-OPT')
ax.plot(Rs, less_l, "s-", label='FedAvg stragglers')
ax.plot(Rs, prox_l, "x-", label='FedProx')
ax.plot(Rs, random_l, "^-", label='CacheFL-random')
handles,labels = ax.get_legend_handles_labels()
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Loss Value')
#ax.set_title('K={:d}, tau={:d}'.format(M, K))
ax.legend(handles, labels, loc='upper right')

ax = fig.add_subplot(212)
ax.plot(Tregular, local_l[0:len(Tregular)], "v-", label='FedAvg')
ax.plot(TcacheFL, delay_l[0:len(TcacheFL)], "o-", label='CacheFL-OPT')
ax.plot(Tless, less_l[0:len(Tless)], "s-", label='FedAvg stragglers')
ax.plot(Tregular, prox_l[0:len(Tregular)], "x-", label='FedProx')
ax.plot(Trandom, random_l[0:len(Trandom)], "^-", label='CacheFL-random')
handles, labels = ax.get_legend_handles_labels()
ax.set_xlabel('Wall-clock Time')
ax.set_ylabel('Loss Value')
#ax.set_title('K={:d}, tau={:d}'.format(M, K))
ax.legend(handles, labels, loc='upper right')

plt.savefig(plotfile)