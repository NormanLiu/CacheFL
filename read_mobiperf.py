import json
import pandas as pd
import random
import numpy as np
import sys

def compute_Tsingle(Kcache, Kserver, Tcomp, Tul, Tdl):
    Tsingle_list = []
    for k in Kcache:
        # Tsingle_list.append(Tcomp[k] + Tul[k])
        # Tsingle_list.append(Tdl[k])
        Tsingle_list.append(min(max(Tcomp[k] + Tul[k], Tdl[k]), max(Tcomp[k] + Tdl[k], Tul[k])))
    for k in Kserver:
        Tsingle_list.append(Tcomp[k] + Tul[k] + Tdl[k])
    return max(Tsingle_list)

def compute_opt_obj(M, x, Tcomp, Tul, Tdl, n_sample_fraction):
    Tsingle_list = []
    T = 0
    for k in range(M):
        Tsingle_list.append((Tcomp[k] + Tul[k] + Tdl[k])*(1-x[k]))
        Tsingle_list.append((Tcomp[k] + Tul[k])*x[k])
        Tsingle_list.append(Tdl[k] * x[k])
        T += n_sample_fraction[k]*x[k]
    Tsingle = max(Tsingle_list)
    return Tsingle * (1+T)

def read_mobiperf(num_tracefile):
    uplink_trace = {}
    downlink_trace = {}

    id_list = []
    common_list = []

    for i in range(num_tracefile):
        with open('measurement' + str(i + 1)) as f:
            data = json.load(f)

        for measurment in range(len(data)):
            measure_dict = data[measurment]
            if measure_dict['type'] == 'tcpthroughput':
                parameters = measure_dict['parameters']
                values = measure_dict['values']
                if 'tcp_speed_results' in values:
                    if parameters['dir_up'] == True:
                        if values['tcp_speed_results'] != '[]':
                            if measure_dict['id'] in id_list:
                                print(measure_dict['id'])
                            else:
                                id_list.append(measure_dict['id'])
    for i in range(num_tracefile):
        with open('measurement' + str(i + 1)) as f:
            data = json.load(f)

        for measurment in range(len(data)):
            measure_dict = data[measurment]
            if measure_dict['type'] == 'tcpthroughput':
                parameters = measure_dict['parameters']
                values = measure_dict['values']
                if 'tcp_speed_results' in values:
                    if parameters['dir_up'] == False:
                        if values['tcp_speed_results'] != '[]':
                            if (measure_dict['id'] in id_list) & (measure_dict['id'] not in common_list):
                                common_list.append(measure_dict['id'])

    for device_id in common_list:
        uplink_trace[device_id] = []
        downlink_trace[device_id] = []

    for i in range(num_tracefile):
        with open('measurement' + str(i + 1)) as f:
            data = json.load(f)

        for measurment in range(len(data)):
            measure_dict = data[measurment]
            if (measure_dict['type'] == 'tcpthroughput') & (measure_dict['id'] in common_list):
                parameters = measure_dict['parameters']
                values = measure_dict['values']
                if 'tcp_speed_results' in values:
                    if parameters['dir_up'] == True:
                        if values['tcp_speed_results'] != '[]':
                            uplink_trace[measure_dict['id']].extend(json.loads(values['tcp_speed_results']))
                    else:
                        if values['tcp_speed_results'] != '[]':
                            downlink_trace[measure_dict['id']].extend(json.loads(values['tcp_speed_results']))
    return [uplink_trace, downlink_trace, common_list]

def optimization(M, R, randomness, n_sample, max_delayed):
    trace = []
    for i in range(5):
        df = pd.read_excel('Failure Models.xlsx', sheet_name=i, nrows=200, usecols=[0, 4, 8, 12, 16])
        df_list = df.values.tolist()
        df_list = [list(i) for i in zip(*df_list)]
        trace.extend(df_list)

    [uplink_trace, downlink_trace, common_list] = read_mobiperf(5)

    Tcomp = []
    Tul = []
    Tdl = []
    if randomness == 'det':
        for k in range(M):
            Tcomp.append(random.choice(trace[k % 25]))
            device_id = common_list[1]
            Tul.append(5.2e3/random.choice(uplink_trace[device_id]))
            Tdl.append(5.2e3/random.choice(downlink_trace[device_id]))
    else:
        for k in range(M):
            Tcomp_list = trace[k % 25]
            Tcomp.append(Tcomp_list[0:R])
            mean_ul = np.random.uniform(10, 20)
            mean_dl = np.random.uniform(10, 20)
            Tul.append(np.random.uniform(mean_ul - 5, mean_ul + 5, R))
            Tdl.append(np.random.uniform(mean_dl - 5, mean_dl + 5, R))

    # Alg.2
    if randomness == 'det':
        Tsingle_server = [x + y + z for x, y, z in zip(Tcomp, Tul, Tdl)]
        Xcache = np.zeros(M)
        total_time_min = compute_opt_obj(M, Xcache, Tcomp, Tul, Tdl, n_sample)
        Xopt = Xcache.copy()
        sort_idx = np.flip(np.argsort(Tsingle_server))
        for k in range(max_delayed):
            Xcache[sort_idx[k]] = 1
            opt_obj_value = compute_opt_obj(M, Xcache, Tcomp, Tul, Tdl, n_sample)
            if opt_obj_value < total_time_min:
                total_time_min = opt_obj_value
                Xopt = Xcache.copy()
        Xcache = np.where(Xopt == 1)
        Xserver = np.where(Xopt == 0)
    return [Xcache[0], Xserver[0], Tcomp, Tdl, Tul]


##################################################################################################################

frac_delay = sys.argv[1] if len(sys.argv) > 1 else 0.5
learning_rate = sys.argv[2] if len(sys.argv) > 2 else 0.01
distribution = sys.argv[3] if len(sys.argv) > 3 else 'het'
dataset = sys.argv[4] if len(sys.argv) > 4 else 'real'
plot_prefix = sys.argv[5] if len(sys.argv) > 5 else 'plots2'
dataset_name = sys.argv[6] if len(sys.argv) > 6 else 'cifar10'
print([frac_delay, learning_rate, distribution, dataset, plot_prefix, dataset_name])


#generate data
C = 10 # number of classes
M = 200
#n_sample = 50*np.random.zipf(2, M) # number of samples on each device follows a power law
#n_sample = 255*np.ones((M,), dtype=int) # balance
# n_sample = np.arange(200,300,2)
n_sample = np.arange(200,600,2)
np.savez('n_sample_large', n_sample = n_sample)
range_samples = [sum(n_sample[0:i]) for i in range(len(n_sample)+1)]
n_sample_fraction = n_sample/sum(n_sample)

[Kcache, Kserver, Tcomp, Tdl, Tul] = optimization(M, 200, 'det', n_sample_fraction, M)
Tsingle_cacheFL = compute_Tsingle(Kcache, Kserver, Tcomp, Tul, Tdl)
Tsingle_regular = compute_Tsingle([], range(M), Tcomp, Tul, Tdl)
Tsingle_less = compute_Tsingle([], Kserver, Tcomp, Tul, Tdl)
T_ratio = Tsingle_cacheFL / Tsingle_regular
Tsingle_less = compute_Tsingle([], Kserver, Tcomp, Tul, Tdl)
Tsingle_random = []
Kcache_list = []
Kserver_list = []
for r in range(20):
    number_delayed = np.random.randint(0, M)
    Kcache_r = np.ndarray.tolist(np.random.choice(M, number_delayed, replace=False))
    Kserver_r = [x for x in range(M) if x not in Kcache_r]
    Kcache_list.append(Kcache_r)
    Kserver_list.append(Kserver_r)
    Tsingle_random.append(compute_Tsingle(Kcache_r, Kserver_r, Tcomp, Tul, Tdl))
Tsingle_k = [Tul[i] + Tdl[i] + Tcomp[i] for i in range(M)]

np.savez('random_delay_large', Kcache = Kcache, Kserver = Kserver, Tsingle_cacheFL = Tsingle_cacheFL, Tsingle_less = Tsingle_less, Tsingle_regular = Tsingle_regular, Tsingle_random = Tsingle_random, Tsingle_k = Tsingle_k)



