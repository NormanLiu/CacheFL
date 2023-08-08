import sys
import numpy as np
import matplotlib
from sklearn.datasets import load_svmlight_file
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
import pandas as pd
import random
import pickle
import os

np.set_printoptions(precision=3, linewidth=240, suppress=True)
np.random.seed(1993)

############################################## Logistic Regression ###############################################
def softmax(x):
    probs = np.exp(np.clip(x, -15, 15))   # n x C or C-dim vector
    if probs.ndim == 1:
        divide = np.sum(probs) # 1
        probs /= divide
    else:
        divide = np.sum(probs, axis=1)  # n x 1
        for i in range(x.shape[0]):
            probs[i] /= divide[i]
    return probs

# features is an [n x d] matrix of features (each row is one data point)
# labels is an [n x C] matrix of labels
# x is an [d x C] matrix of parameters
def mlogit_loss(x, features, labels):
    Loss_val = 0  # define loss
    n = features.shape[0]
    for i in range(n):
        probs = softmax(np.dot(np.transpose(x), features[i])) # C-dimensional vector of probs for each class
        Loss_val -= np.dot(labels[i], np.log(1e-12 + np.transpose(probs)))
    return (1./n) * Loss_val

# return an [d x C] matrix of the gradients
def mlogit_loss_full_gradient(x, features, labels):
    grads = np.zeros_like(x)
    probs = softmax(np.dot(features, x))  # n x C
    for i in range(x.shape[1]):
        grads[:, i] -= np.dot(np.transpose(features), labels[:, i] - probs[:,i]) # d-dimensional vector of the grads of class i
    return grads / features.shape[0]

def mlogit_loss_stochastic_gradient(x, features, labels, minibatch_size):
    grads = np.zeros_like(x)
    idxs = np.random.randint(0, features.shape[0], minibatch_size)
    fts = features[idxs, :]
    probs = softmax(np.dot(fts, x))  # minibatch_size x C
    for i in range(x.shape[1]):
        grads[:, i] -= np.dot(np.transpose(fts), labels[idxs, i] - probs[:, i]) # d-dimensional vector of the grads of class i
    return grads / minibatch_size

def fedprox_loss_stochastic_gradient(x, features, labels, minibatch_size, xs):
    grads = np.zeros_like(x)
    idxs = np.random.randint(0, features.shape[0], minibatch_size)
    fts = features[idxs, :]
    probs = softmax(np.dot(fts, x))  # minibatch_size x C
    for i in range(x.shape[1]):
        grads[:, i] -= np.dot(np.transpose(fts), labels[idxs, i] - probs[:, i])  # d-dimensional vector of the grads of class i
    return grads / minibatch_size + 0.1*(x-xs)


##################################################################################################################

def one_inner_outer_iteration(x_start, Kset, K, stepsize):
    grads = np.zeros_like(x_start)
    for m in Kset:
        x = x_start.copy()
        for _ in range(K):
            g = mlogit_loss_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 10)
            grads += g * n_sample_fraction[m]
            x -= stepsize * g
    return grads

def one_inner_outer_iteration2(x_start_latest, x_start_delayed, Kcache, Kserver, K, stepsize):
    grads = np.zeros_like(x_start_latest)
    for m in Kserver:
        x = x_start_latest.copy()
        for _ in range(K):
            g = mlogit_loss_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 10)
            grads += g * n_sample_fraction[m]
            x -= stepsize * g
    for m in Kcache:
        x = x_start_delayed.copy()
        for _ in range(K):
            g = mlogit_loss_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 10)
            grads += g * n_sample_fraction[m]
            x -= stepsize * g
#    grads
    return grads

def one_inner_outer_iteration3(x_start, M, K, stepsize):
    grads = np.zeros_like(x_start)
    for m in range(M):
        x = x_start.copy()
        for _ in range(K):
            g = fedprox_loss_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 1, x_start)
            grads += g * n_sample_fraction[m]
            x -= stepsize * g
    """
    for m in range(M // 2):
        x = x_start.copy()
        for _ in range(int(K)):
            g = fedprox_loss_stochastic_gradient(x, features[m*datasize:(m+1)*datasize,:], labels[m*datasize:(m+1)*datasize], 10, x_start)
            grads += g / M
            x -= stepsize * g
    for m in range(M // 2):
        x = x_start.copy()
        m += M // 2
        for _ in range(K):
            g = fedprox_loss_stochastic_gradient(x, features[m*datasize:(m+1)*datasize,:], labels[m*datasize:(m+1)*datasize], 10, x_start)
            grads += g / M
            x -= stepsize * g
    """
    return grads

def inner_outer_sgd(x0_len, Kset, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
    losses = []
    accuracies = []
    iterates = [np.zeros((x0_len, C))]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        direction = one_inner_outer_iteration(iterates[-1], Kset, K, inner_stepsize)
        iterates.append(iterates[-1] - outer_stepsize * direction)
        if (r+1) % loss_freq == 0:
            losses.append(objective_value(np.average(iterates,axis=0)))
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
            accuracies.append(sample_accuracy(np.average(iterates, axis=0)))
    print('')
    return [losses, accuracies]

def inner_outer_sgd2(x0_len, Kcache, Kserver, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=3):
    losses = []
    accuracies = []
    iterates = [np.zeros((x0_len, C))]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        if r>1:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-2], Kcache, Kserver, K, inner_stepsize)
        else:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-1], Kcache, Kserver, K, inner_stepsize)
        iterates.append(iterates[-1] - outer_stepsize * direction)
        if (r+1) % loss_freq == 0:
            losses.append(objective_value(np.average(iterates,axis=0)))
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
            accuracies.append(sample_accuracy(np.average(iterates, axis=0)))
            #print('Iteration: {:d}/{:d}   Accuracy: {:f}                 \r'.format(r + 1, R, accuracies[-1]), end='')
    print('')
    return [losses, accuracies]



def inner_outer_sgd3(x0_len, M, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
    losses = []
    accuracies = []
    iterates = [np.zeros((x0_len, C))]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        direction = one_inner_outer_iteration3(iterates[-1], M, K, inner_stepsize)
        iterates.append(iterates[-1] - outer_stepsize * direction)
        if (r+1) % loss_freq == 0:
            losses.append(objective_value(np.average(iterates,axis=0)))
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
            accuracies.append(sample_accuracy(np.average(iterates,axis=0)))
    print('')
    return [losses, accuracies]

def inner_outer_sgd4(x0_len, M, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
    losses = []
    accuracies = []
    iterates = [np.zeros((x0_len, C))]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        if r>1:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-2], Kcache_list[r//10], Kserver_list[r//10], K, inner_stepsize)
        else:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-1], Kcache_list[r//10], Kserver_list[r//10], K, inner_stepsize)
        iterates.append(iterates[-1] - outer_stepsize * direction)
        if (r+1) % loss_freq == 0:
            losses.append(objective_value(np.average(iterates,axis=0)))
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
            accuracies.append(sample_accuracy(np.average(iterates, axis=0)))
    print('')
    return [losses, accuracies]

def local_sgd(x0_len, Kset, K, R, stepsize, loss_freq):
    return inner_outer_sgd(x0_len, Kset, K, R, stepsize, stepsize, loss_freq)

def local_sgd_delayed(x0_len, Kcache, Kserver, K, R, stepsize, loss_freq):
    return inner_outer_sgd2(x0_len, Kcache, Kserver, K, R, stepsize, stepsize, loss_freq)

def local_sgd_delayed_random(x0_len, M, K, R, stepsize, loss_freq):
    return inner_outer_sgd4(x0_len, M, K, R, stepsize, stepsize, loss_freq)

def fedprox(x0_len, M, K, R, stepsize, loss_freq):
    return inner_outer_sgd3(x0_len, M, K, R, stepsize, stepsize, loss_freq)

def minibatch_sgd(x0_len, T, batchsize, stepsize, loss_freq, avg_window=8):
    losses = []
    iterates = [np.zeros((x0_len, C))]
    for t in range(T):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        iterates.append(iterates[-1] - stepsize * objective_stochastic_gradient(iterates[-1], batchsize))
        if (t+1) % loss_freq == 0:
            losses.append(objective_value(np.average(iterates,axis=0)))
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(t+1,T,losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
    print('')
    return losses

def gradient_descent(x0_len, T = 2000):
    n_stepsizes = 10
    tt_stepsizes = [np.exp(exponent) for exponent in np.linspace(-5, 0, n_stepsizes)]
    losses = []
    for stepsize in tt_stepsizes:
        x = np.zeros((x0_len, C))
        for t in range(T):
            x -= stepsize * objective_full_gradient(x)
            stepsize *= 0.995
        losses.append(objective_value(x))
    return min(losses)

"""
def newtons_method(x0_len, max_iter=1000, tol=1e-6):
    x = np.zeros(x0_len)
    stepsize = 0.5
    for t in range(max_iter):
        gradient = objective_full_gradient(x)
        hessian = objective_hessian(x)
        update_direction = np.linalg.solve(hessian, gradient)
        x -= stepsize * update_direction
        newtons_decrement = np.sqrt(np.dot(gradient, update_direction))
        if newtons_decrement <= tol:
            print("Newton's method converged after {:d} iterations".format(t+1))
            return objective_value(x)
    print("Warning: Newton's method failed to converge")
    return objective_value(x)
"""

# X_test is an [n x d] matrix of features (each row is one data point)
# y_test is an [n x C] matrix of labels
# w is an [d x C] matrix of parameters
def model_accoracy(X_test, y_test, w):
    #X_test = np.insert(X_test, 0, 1, axis=1)  # add constant
    predictions = []  # define a prediction list
    for i in range(X_test.shape[0]):  # iterate n samples
        prob = softmax(np.dot(X_test[i], w))  # softmax
        predict = np.argmax(prob)  # find the index with maximum probability
        predictions.append(predict)  # add the final prediction to the list

    accuracy = np.count_nonzero(pd.Series(predictions) == pd.Series(y_test)) / len(predictions)
    return accuracy

def compute_Tsingle(Kcache, Kserver, Tcomp, Tul, Tdl):
    Tsingle_list = []
    for k in Kcache:
        Tsingle_list.append(Tcomp[k] + Tul[k])
        Tsingle_list.append(Tdl[k])
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

def optimization(M, R, randomness, n_sample):
    trace = []
    for i in range(5):
        df = pd.read_excel('Failure Models.xlsx', sheet_name=i, nrows=200, usecols=[0, 4, 8, 12, 16])
        df_list = df.values.tolist()
        df_list = [list(i) for i in zip(*df_list)]
        trace.extend(df_list)
    Tcomp = []
    Tul = []
    Tdl = []
    if randomness == 'det':
        for k in range(M):
            Tcomp.append(random.choice(trace[k % 25]))
            mean_ul = np.random.uniform(10, 20)
            mean_dl = np.random.uniform(10, 20)
            Tul.append(5*np.random.uniform(mean_dl - 5, mean_dl + 5))
            Tdl.append(5*np.random.uniform(mean_dl - 5, mean_dl + 5))
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
        for k in range(M):
            Xcache[sort_idx[k]] = 1
            opt_obj_value = compute_opt_obj(M, Xcache, Tcomp, Tul, Tdl, n_sample)
            if opt_obj_value < total_time_min:
                total_time_min = opt_obj_value
                Xopt = Xcache.copy()
        Xcache = np.where(Xopt == 1)
        Xserver = np.where(Xopt == 0)
    return [Xcache[0], Xserver[0], Tcomp, Tdl, Tul]

##################################################################################################################

def experiment(M,K,R,plotfile,savearray):
    loss_freq = 5
    n_reps = 2

    l0 = objective_value(np.zeros((x0_len, C))) - fstar
    a0 = sample_accuracy(np.zeros((x0_len, C)))
    delay_l_list = []
    delay_a_list = []
    Kcache = []
    Kserver = list(range(M))

    print('Doing Partial Delayed Local SGD:   {:d}/{:d}'.format(0, 5))
    delay_results = np.zeros((R // loss_freq, len(tt_stepsizes)))
    delay_results_acc = np.zeros((R // loss_freq, len(tt_stepsizes)))
    for i, stepsize in enumerate(tt_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(tt_stepsizes)))
        for rep in range(n_reps):
            [losses, accuracies] = local_sgd_delayed(x0_len, Kcache, Kserver, K, R, stepsize, loss_freq)
            delay_results[:, i] += (losses - fstar) / n_reps
            delay_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps
    delay_l_list.append(np.concatenate([[l0], np.min(delay_results, axis=1)]))
    delay_a_list.append(np.concatenate([[a0], np.max(delay_results_acc, axis=1)]))

    Tsingle_server = [x + y + z for x, y, z in zip(Tcomp, Tul, Tdl)]
    sort_idx = np.flip(np.argsort(Tsingle_server))
    for k in range(M):
        Kcache.append(sort_idx[k])
        Kserver.remove(sort_idx[k])
        if (k+1) % 10 == 0:
            print('Doing Partial Delayed Local SGD:   {:d}/{:d}'.format((k+1)%10, 5))
            delay_results = np.zeros((R // loss_freq, len(tt_stepsizes)))
            delay_results_acc = np.zeros((R // loss_freq, len(tt_stepsizes)))
            for i, stepsize in enumerate(tt_stepsizes):
                print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(tt_stepsizes)))
                for rep in range(n_reps):
                    [losses, accuracies] = local_sgd_delayed(x0_len, Kcache, Kserver, K, R, stepsize, loss_freq)
                    delay_results[:, i] += (losses - fstar) / n_reps
                    delay_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps
            delay_l_list.append(np.concatenate([[l0], np.min(delay_results, axis=1)]))
            delay_a_list.append(np.concatenate([[a0], np.max(delay_results_acc, axis=1)]))



    np.savez(savearray, delay_l_list, delay_a_list)

    Rs = [0] + list(range(loss_freq, R+1, loss_freq))

    line_type = ["v-", "o-", "s-", "x-", "^-", "|-"]

    # plot results
    fig = plt.figure(figsize=(12.8, 4.8))
    ax = fig.add_subplot(121)
    for k in range(5 + 1):
        ax.plot(Rs, delay_l_list[k], line_type[k], label='|Kcache|={:d}'.format(k * 10))
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Loss Value')
    ax.legend(handles, labels, loc='upper right')

    ax = fig.add_subplot(122)
    for k in range(5 + 1):
        ax.plot(Rs, delay_a_list[k], line_type[k], label='|Kcache|={:d}'.format(k * 10))
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Accuracy')
    ax.legend(handles, labels, loc='lower right')


    plt.savefig(plotfile)


##################################################################################################################

frac_delay = sys.argv[1] if len(sys.argv) > 1 else 0.5
learning_rate = sys.argv[2] if len(sys.argv) > 2 else 0.01
distribution = sys.argv[3] if len(sys.argv) > 3 else 'het'
dataset = sys.argv[4] if len(sys.argv) > 4 else 'real'
plot_prefix = sys.argv[5] if len(sys.argv) > 5 else 'plots2'
dataset_name = sys.argv[6] if len(sys.argv) > 6 else 'synthetic'
print([frac_delay, learning_rate, distribution, dataset, plot_prefix, dataset_name])


#generate data
C = 10 # number of classes
# number of samples on each device follows a power law
M = 50
n_sample = 5*np.random.zipf(2, M)
range_samples = [sum(n_sample[0:i]) for i in range(len(n_sample)+1)]
n_sample_fraction = n_sample/sum(n_sample)

if dataset == 'real':
    data = load_svmlight_file(dataset_name)
    features, labels_array = data[0].toarray(), data[1]
    if dataset_name == 'cifar10':
        features = features/255
    if distribution == 'iid':
        features = features[0:range_samples[-1]]
        labels_array = labels_array[0:range_samples[-1]]
    elif distribution == 'het':
        samples_idx = []
        for k in range(M):
            classes = np.random.choice(10, 2, replace=False)
            candidate1 = np.where(labels_array == classes[0])
            candidate2 = np.where(labels_array == classes[1])
            candidate = np.concatenate([candidate1[0], candidate2[0]])
            samples_idx += np.ndarray.tolist(np.random.choice(candidate, n_sample[k], replace=False))
        features = features[samples_idx]
        labels_array = labels_array[samples_idx]
    N = features.shape[0]
    labels = np.zeros((N, C))
    for i in range(N):
        labels[i, int(labels_array[i])] = 1
    features_t, labels_t = features, labels_array
    #data_t = load_svmlight_file(dataset_name)
    #features_t, labels_t = data_t[0].toarray(), data_t[1]
else:
    # iid case: same W,b~N(0,1) on all devices, all samples ~N(v, \Sigma)
    if distribution == 'iid':
        features = np.zeros((sum(n_sample), 60))
        labels_array = np.zeros(sum(n_sample))
        W = np.random.randn(C, 60)
        b = 0*np.random.randn(C)
        Sigma = [(j + 1) ** (-1.2) for j in range(60)]
        for i in range(sum(n_sample)):
            features[i] = np.random.normal(0, Sigma)
            prob = softmax(np.dot(W, features[i]) + b)
            observe = np.argmax(prob)
            predict_prob = softmax(np.dot(W, features[i]))
            predict = np.argmax(predict_prob)
            labels_array[i] = np.argmax(softmax(np.dot(W, features[i]) + b))
    # het case: Synthetic(alpha, beta)
    elif distribution == 'het':
        features = []
        labels_array = []
        alpha, beta = 1, 1
        for k in range(M):
            Bk = np.random.normal(0, beta, 60)
            vk = np.random.normal(Bk, [1 for i in range(60)])
            uk = np.random.normal(0, alpha)
            bk = np.random.normal(uk, 1, C)
            Wk = np.random.normal(uk, 1, (C, 60))
            Sigma = [(j + 1) ** (-1.2) for j in range(60)]
            for i in range(n_sample[k]):
                xk = np.random.normal(vk, Sigma)
                features.append(xk)
                yk = np.argmax(softmax(np.dot(Wk, xk) + bk))
                labels_array.append(yk)
        features = np.array(features)
        labels_array = np.array(labels_array)

    N = features.shape[0]
    labels = np.zeros((N, C))
    for i in range(N):
        labels[i, int(labels_array[i])] = 1
    features_t, labels_t = features, labels_array

# generate stepsizes
if learning_rate == 'grid_search':
    n_stepsizes = 8
    tt_stepsizes = [np.exp(exponent) for exponent in np.linspace(-5, -1, n_stepsizes)]
    lg_stepsizes = [np.exp(exponent) for exponent in np.linspace(-5, -1, n_stepsizes)]
    lc_stepsizes = [np.exp(exponent) for exponent in np.linspace(-5, -1, n_stepsizes)]
else:
    tt_stepsizes = [float(learning_rate)]
    lg_stepsizes = [float(learning_rate)]
    lc_stepsizes = [float(learning_rate)]

npzfile = np.load('random_delay.npz')
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


[Kcache, Kserver, Tcomp, Tdl, Tul] = optimization(M, 200, 'det', n_sample_fraction)
"""
# generate client partition
[Kcache, Kserver, Tcomp, Tdl, Tul] = optimization(M, 200, 'det', n_sample_fraction)
Tsingle_cacheFL = compute_Tsingle(Kcache, Kserver, Tcomp, Tul, Tdl)
Tsingle_regular = compute_Tsingle([], range(M), Tcomp, Tul, Tdl)
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
#Kcache = np.ndarray.tolist(np.random.choice(M, int(M*float(frac_delay)), replace=False))
#Kcache.sort()
#Kserver = [x for x in range(M) if x not in Kcache]

np.savez('random_delay', Kcache = Kcache, Kserver = Kserver, Tsingle_cacheFL = Tsingle_cacheFL, Tsingle_less = Tsingle_less, Tsingle_regular = Tsingle_regular, Tsingle_random = Tsingle_random)
with open("Kcache_list", "wb") as fp:
    pickle.dump(Kcache_list, fp)
with open("Kserver_list", "wb") as fp:
    pickle.dump(Kserver_list, fp)
"""

loss_function = 'binary logistic loss'
objective_value = lambda x: mlogit_loss(x, features, labels) #+ 0.05*np.linalg.norm(x)**2
objective_full_gradient = lambda x: mlogit_loss_full_gradient(x, features, labels) #+ 0.1*x
objective_stochastic_gradient = lambda x, minibatch_size: mlogit_loss_stochastic_gradient(x, features, labels, minibatch_size) #+ 0.1*x
fedprox_stochastic_gradient = lambda x, xg, minibatch_size: fedprox_loss_stochastic_gradient(x, features, labels, minibatch_size) + 0.1*(x-xg)
sample_accuracy = lambda w: model_accoracy(features_t, labels_t, w)

x0_len = features.shape[1]
#fstar = gradient_descent(x0_len)
fstar = np.float64(0)
print('Fstar = {:.5f}'.format(fstar))

#experiment(M=10,K=5,R=200,plotfile= plot_prefix + '/smallerM-smallK-' + distribution + '.png',savearray=plot_prefix + '/smallerM-smallK-' + distribution)
experiment(M,K=5,R=400,plotfile= plot_prefix + '/smallM-smallK-' + distribution + dataset_name + '.png',savearray=plot_prefix + '/smallM-smallK-' + distribution + dataset_name)
#experiment(M=500,K=5,R=200,plotfile= plot_prefix + '/bigM-smallK-' + distribution + '.png',savearray=plot_prefix + '/bigM-smallK-' + distribution)
#experiment(M=50,K=40,R=200,plotfile= plot_prefix + '/smallM-bigK-' + distribution + '.png',savearray=plot_prefix + '/smallM-bigK-' + distribution)
#experiment(M=500,K=40,R=100,plotfile='plots/bigM-bigK-het.png',savearray='plots/bigM-bigK-het')
#experiment(M=50,K=200,R=100,plotfile='plots/smallM-biggerK-het.png',savearray='plots/smallM-biggerK-het')
