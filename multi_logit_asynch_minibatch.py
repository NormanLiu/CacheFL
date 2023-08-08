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
#np.random.seed(1993)
random.seed(1993)
np.random.seed(20180407)

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
def one_single_client_iteration(x_start, action_client, K, stepsize):
    grads = np.zeros_like(x_start)
    x = x_start.copy()
    for _ in range(K):
        g = mlogit_loss_stochastic_gradient(x, features[range_samples[action_client]:range_samples[action_client+1],:], labels[range_samples[action_client]:range_samples[action_client+1]], 5)
        #grads += g * n_sample_fraction[action_client]
        grads += g
        x -= stepsize * g
    return [grads, x]

def one_inner_outer_iteration(x_start, Kset, K, stepsize):
    grads = np.zeros_like(x_start)
    for m in Kset:
        x = x_start.copy()
        for _ in range(K):
            g = mlogit_loss_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 5)
            grads += g * n_sample_fraction[m]
            x -= stepsize * g
    return grads

def one_inner_outer_iteration2(x_start_latest, x_start_delayed, Kcache, Kserver, K, stepsize):
    grads = np.zeros_like(x_start_latest)
    for m in Kserver:
        x = x_start_latest.copy()
        for _ in range(K):
            g = mlogit_loss_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 5)
            grads += g * n_sample_fraction[m]
            x -= stepsize * g
    for m in Kcache:
        x = x_start_delayed.copy()
        for _ in range(K):
            g = mlogit_loss_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 5)
            grads += g * n_sample_fraction[m]
            x -= stepsize * g
#    grads
    return grads

def one_inner_outer_iteration3(x_start, M, K, stepsize):
    grads = np.zeros_like(x_start)
    for m in range(M):
        x = x_start.copy()
        for _ in range(K):
            g = fedprox_loss_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 5, x_start)
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

def inner_outer_sgd5(x0_len, Kset, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
    losses = []
    accuracies = []
    iterates = [np.zeros((x0_len, C))]
    initial_k = [iterates[0] for _ in range(M)]
    initial_iter_k = [0 for _ in range(M)]
    for r in range(R * M * 2):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        action_client = clientAction_list[r]
        [direction, model] = one_single_client_iteration(initial_k[action_client], action_client, K, inner_stepsize)
        if agg_scheme == 0:
            iterates.append(iterates[-1] - outer_stepsize * direction)
        elif agg_scheme == 1:
            iterates.append(0.5 * iterates[-1] + 0.5 * model)
        else:
            alpha = 0.5/(r - initial_iter_k[action_client] + 1)
            iterates.append((1-alpha)*iterates[-1] + alpha*model)
        initial_k[action_client] = iterates[-1]
        initial_iter_k[action_client] = r + 1
        if (r+1) % (loss_freq * M) == 0:
            losses.append(objective_value(np.average(iterates,axis=0)))
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format((r+1)//M, R*2, losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
            accuracies.append(sample_accuracy(np.average(iterates, axis=0)))
    print('')
    return [losses, accuracies]

def inner_outer_sgd2(x0_len, Kcache, Kserver, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
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
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-2], Kcache_list[r//3%20], Kserver_list[r//3%20], K, inner_stepsize)
        else:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-1], Kcache_list[r//3%20], Kserver_list[r//3%20], K, inner_stepsize)
        iterates.append(iterates[-1] - outer_stepsize * direction)
        if (r+1) % loss_freq == 0:
            losses.append(objective_value(np.average(iterates,axis=0)))
            #print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
            accuracies.append(sample_accuracy(np.average(iterates, axis=0)))
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r + 1, R, accuracies[-1]), end='')
    print('')
    return [losses, accuracies]

def local_sgd(x0_len, Kset, K, R, stepsize, loss_freq):
    return inner_outer_sgd(x0_len, Kset, K, R, stepsize, stepsize, loss_freq)

def local_sgd_asynch(x0_len, Kset, K, R, stepsize, loss_freq):
    return inner_outer_sgd5(x0_len, Kset, K, R, stepsize, stepsize, loss_freq)

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
    n_stepsizes = 5
    tt_stepsizes = [np.exp(exponent) for exponent in np.linspace(-5, 0, n_stepsizes)]
    losses = []
    tt_stepsizes = [0.5]
    for stepsize in tt_stepsizes:
        x = np.zeros((x0_len, C))
        for t in range(T):
            x -= stepsize * objective_full_gradient(x)
            stepsize *= 0.997
            if (t+1)%5 == 0:
                print(sample_accuracy(x))
        losses.append(objective_value(x))
        print(sample_accuracy(x))
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

def optimization(M, R, randomness, n_sample, max_delayed):
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
            #mean_ul = np.random.uniform(10, 20)
            #mean_dl = np.random.uniform(10, 20)
            mean_dl = np.random.normal(15, 10)
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

def experiment(M,K,R,plotfile,savearray):
    loss_freq = 2
    n_reps = 2

    print('Doing asynch FedAvg...')
    asynch_results = np.zeros((R * 2//loss_freq, len(lc_stepsizes)))
    asynch_results_acc = np.zeros((R * 2 // loss_freq, len(lc_stepsizes)))
    for i,stepsize in enumerate(lc_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(lc_stepsizes)))
        for rep in range(n_reps):
            [losses, accuracies] = local_sgd_asynch(x0_len, range(M), K, R, stepsize, loss_freq)
            asynch_results[:, i] += (losses - fstar) / n_reps
            asynch_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps

    print('Doing Random Delayed Local SGD...')
    random_results = np.zeros((R // loss_freq, len(tt_stepsizes)))
    random_results_acc = np.zeros((R // loss_freq, len(tt_stepsizes)))
    for i, stepsize in enumerate(tt_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(tt_stepsizes)))
        for rep in range(n_reps):
            [losses, accuracies] = local_sgd_delayed_random(x0_len, M, K, R, stepsize, loss_freq)
            random_results[:, i] += (losses - fstar) / n_reps
            random_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps

    print('Doing Partial Delayed Local SGD...')
    delay_results = np.zeros((R // loss_freq, len(tt_stepsizes)))
    delay_results_acc = np.zeros((R // loss_freq, len(tt_stepsizes)))
    for i, stepsize in enumerate(tt_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(tt_stepsizes)))
        for rep in range(n_reps):
            [losses, accuracies] = local_sgd_delayed(x0_len, Kcache, Kserver, K, R, stepsize, loss_freq)
            delay_results[:, i] += (losses - fstar) / n_reps
            delay_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps

    print('Doing FedProx...')
    prox_results = np.zeros((R // loss_freq, len(lc_stepsizes)))
    prox_results_acc = np.zeros((R // loss_freq, len(lc_stepsizes)))
    for i, stepsize in enumerate(lc_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(lc_stepsizes)))
        for rep in range(n_reps):
            [losses, accuracies] = fedprox(x0_len, M, K, R, stepsize, loss_freq)
            prox_results[:, i] += (losses - fstar) / n_reps
            prox_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps

    print('Doing Local SGD with less workers...')
    less_results = np.zeros((R//loss_freq, len(lg_stepsizes)))
    less_results_acc = np.zeros((R // loss_freq, len(lg_stepsizes)))
    for i,stepsize in enumerate(lg_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(lg_stepsizes)))
        for rep in range(n_reps):
            [losses, accuracies] = local_sgd(x0_len, Kserver, K, R, stepsize, loss_freq)
            less_results[:, i] += (losses - fstar) / n_reps
            less_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps

    print('Doing Local SGD...')
    local_results = np.zeros((R//loss_freq, len(lc_stepsizes)))
    local_results_acc = np.zeros((R // loss_freq, len(lc_stepsizes)))
    for i,stepsize in enumerate(lc_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(lc_stepsizes)))
        for rep in range(n_reps):
            [losses, accuracies] = local_sgd(x0_len, range(M), K, R, stepsize, loss_freq)
            local_results[:, i] += (losses - fstar) / n_reps
            local_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps



    l0 = objective_value(np.zeros((x0_len, C)))-fstar
    local_l = np.concatenate([[l0], np.min(local_results, axis=1)])
    delay_l = np.concatenate([[l0], np.min(delay_results, axis=1)])
    less_l = np.concatenate([[l0], np.min(less_results, axis=1)])
    prox_l = np.concatenate([[l0], np.min(prox_results, axis=1)])
    random_l = np.concatenate([[l0], np.min(random_results, axis=1)])
    asynch_l = np.concatenate([[l0], np.min(asynch_results, axis=1)])

    a0 = sample_accuracy(np.zeros((x0_len, C)))
    local_a = np.concatenate([[a0], np.max(local_results_acc, axis=1)])
    delay_a = np.concatenate([[a0], np.max(delay_results_acc, axis=1)])
    less_a = np.concatenate([[a0], np.max(less_results_acc, axis=1)])
    prox_a = np.concatenate([[a0], np.max(prox_results_acc, axis=1)])
    random_a = np.concatenate([[a0], np.max(random_results_acc, axis=1)])
    asynch_a = np.concatenate([[a0], np.max(asynch_results_acc, axis=1)])

    #np.savez(savearray, local_l, delay_l, less_l, prox_l)
    np.savez(savearray, local_l=local_l, delay_l=delay_l, less_l=less_l, prox_l=prox_l, random_l=random_l,
             local_a=local_a, delay_a=delay_a, less_a=less_a, prox_a=prox_a, random_a=random_a, asynch_l=asynch_l,
             asynch_a=asynch_a)

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


    # plot results
    fig = plt.figure(figsize=(6.4, 14.4))
    ax = fig.add_subplot(311)
    ax.plot(Rs, local_l, "v-", label='FedAvg')
    ax.plot(Rs, delay_l, "o-", label='CacheFL-OPT')
    ax.plot(Rs, less_l, "s-", label='FedAvg stragglers')
    ax.plot(Rs, prox_l, "x-", label='FedProx')
    ax.plot(Rs, random_l, "^-", label='CacheFL-random')
    handles,labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Training Loss')
    #ax.set_title('K={:d}, tau={:d}'.format(M, K))
    ax.legend(handles, labels, loc='upper right')

    ax = fig.add_subplot(312)
    ax.plot(Tregular, local_l[0:len(Tregular)], "v-", label='FedAvg')
    ax.plot(TcacheFL, delay_l[0:len(TcacheFL)], "o-", label='CacheFL-OPT')
    ax.plot(Tless, less_l[0:len(Tless)], "s-", label='FedAvg stragglers')
    ax.plot(Tregular, prox_l[0:len(Tregular)], "x-", label='FedProx')
    ax.plot(Trandom, random_l[0:len(Trandom)], "^-", label='CacheFL-random')
    ax.plot(Tasynch, asynch_l[0:len(Tasynch)], "--", label='FedAvg asynch')
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Wall-clock Time')
    ax.set_ylabel('Training Loss')
    #ax.set_title('K={:d}, tau={:d}'.format(M, K))
    ax.legend(handles, labels, loc='upper right')

    ax = fig.add_subplot(313)
    ax.plot(Tregular, local_a[0:len(Tregular)], "v-", label='FedAvg')
    ax.plot(TcacheFL, delay_a[0:len(TcacheFL)], "o-", label='CacheFL-OPT')
    ax.plot(Tless, less_a[0:len(Tless)], "s-", label='FedAvg stragglers')
    ax.plot(Tregular, prox_a[0:len(Tregular)], "x-", label='FedProx')
    ax.plot(Trandom, random_a[0:len(Trandom)], "^-", label='CacheFL-random')
    ax.plot(Tasynch, asynch_a[0:len(Tasynch)], "--", label='FedAvg asynch')
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Wall-clock time')
    ax.set_ylabel('Test Accuracy')
    # ax.set_title('K={:d}, tau={:d}'.format(M, K))
    ax.legend(handles, labels, loc='lower right')

    """
    ax = fig.add_subplot(222)
    ax.plot(Rs, local_a, "v-", label='FedAvg')
    ax.plot(Rs, delay_a, "o-", label='CacheFL-OPT')
    ax.plot(Rs, less_a, "s-", label='FedAvg stragglers')
    ax.plot(Rs, prox_a, "x-", label='FedProx')
    ax.plot(Rs, random_a, "^-", label='CacheFL-random')
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Round of Communication')
    ax.set_ylabel('Accuracy')
    ax.set_title('K={:d}, tau={:d}'.format(M, K))
    ax.legend(handles, labels, loc='lower right')

    ax = fig.add_subplot(224)
    ax.plot(Tregular, local_a, "v-", label='FedAvg')
    ax.plot(TcacheFL, delay_a, "o-", label='CacheFL-OPT')
    ax.plot(Tless, less_a, "s-", label='FedAvg stragglers')
    ax.plot(Tregular, prox_a, "x-", label='FedProx')
    ax.plot(Trandom, random_a, "^-", label='CacheFL-random')
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Wall clock time')
    ax.set_ylabel('Accuracy')
    ax.set_title('K={:d}, tau={:d}'.format(M, K))
    ax.legend(handles, labels, loc='lower right')
    """
    plt.savefig(plotfile)


##################################################################################################################

frac_delay = sys.argv[1] if len(sys.argv) > 1 else 0.5
learning_rate = sys.argv[2] if len(sys.argv) > 2 else 0.01
distribution = sys.argv[3] if len(sys.argv) > 3 else 'het'
dataset = sys.argv[4] if len(sys.argv) > 4 else 'real'
plot_prefix = sys.argv[5] if len(sys.argv) > 5 else 'plots2'
dataset_name = sys.argv[6] if len(sys.argv) > 6 else 'mnist.scale'
agg_scheme = sys.argv[7] if len(sys.argv) > 7 else 2
print([frac_delay, learning_rate, distribution, dataset, plot_prefix, dataset_name])

agg_scheme = int(agg_scheme)

#generate data
C = 10 # number of classes
# number of samples on each device follows a power law
M = 50
#n_sample = 50*np.random.zipf(2, M)
#np.savez('n_sample', n_sample = n_sample)
if dataset_name == 'mnist.scale':
    #npzfile = np.load('n_sample_mnist.npz')
    npzfile = np.load('n_sample_balance_less.npz')
elif dataset_name == 'cifar10':
    npzfile = np.load('n_sample_cifar10.npz')
n_sample = npzfile['n_sample']
#n_sample = n_sample * 2 # remember to remove this line
range_samples = [sum(n_sample[0:i]) for i in range(len(n_sample)+1)]
n_sample_fraction = n_sample/sum(n_sample)

#n_sample = np.int_(n_sample * 1.5) # remember to remove this line

"""
# generate client partition
[Kcache, Kserver, Tcomp, Tdl, Tul] = optimization(M, 200, 'det', n_sample_fraction, 5)
Tsingle_cacheFL = compute_Tsingle(Kcache, Kserver, Tcomp, Tul, Tdl)
Tsingle_regular = compute_Tsingle([], range(M), Tcomp, Tul, Tdl)
T_ratio = Tsingle_cacheFL/Tsingle_regular
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

np.savez('random_delay_lesser', Kcache = Kcache, Kserver = Kserver, Tsingle_cacheFL = Tsingle_cacheFL, Tsingle_less = Tsingle_less, Tsingle_regular = Tsingle_regular, Tsingle_random = Tsingle_random)
"""

if dataset == 'real':
    data = load_svmlight_file(dataset_name)
    features, labels_array = data[0].toarray(), data[1]
    if dataset_name == 'cifar10':
        features = features/255
    features_t, labels_t = features, labels_array
    if distribution == 'iid':
        features = np.concatenate((features, features))
        labels_array = np.concatenate((labels_array, labels_array))
        features = features[0:range_samples[-1]]
        labels_array = labels_array[0:range_samples[-1]]
    elif distribution == 'het':
        samples_idx = []
        for k in range(M):
            # if dataset_name == 'mnist.scale':
            #     classes = np.random.choice(10, 1, replace=False)
            #     candidate1 = np.where(labels_array == classes[0])
            #     samples_idx += np.ndarray.tolist(np.random.choice(candidate1[0], n_sample[k], replace=True))
            # else:
            #     classes = np.random.choice(10, 2, replace=False)
            #     candidate1 = np.where(labels_array == classes[0])
            #     candidate2 = np.where(labels_array == classes[1])
            #     candidate = np.concatenate([candidate1[0], candidate2[0]])
            #     samples_idx += np.ndarray.tolist(np.random.choice(candidate, n_sample[k], replace=True))
            classes = np.random.choice(10, 2, replace=False)
            candidate1 = np.where(labels_array == classes[0])
            candidate2 = np.where(labels_array == classes[1])
            candidate = np.concatenate([candidate1[0], candidate2[0]])
            samples_idx += np.ndarray.tolist(np.random.choice(candidate, n_sample[k], replace=False))
        features = features[samples_idx]
        labels_array = labels_array[samples_idx]
    #features_t, labels_t = features, labels_array
    N = features.shape[0]
    labels = np.zeros((N, C))
    for i in range(N):
        labels[i, int(labels_array[i])] = 1
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
    if dataset_name == 'mnist.scale':
        data = load_svmlight_file(dataset_name + '.t')
        features_t, labels_t = data[0].toarray(), data[1]
    else:
        features_t, labels_t = features, labels_array


def countFreq(arr, n):
    # Mark all array elements as not visited
    visited = [False for i in range(n)]
    freq_list = []

    # Traverse through array elements
    # and count frequencies
    for i in range(n):

        # Skip this element if already
        # processed
        if (visited[i] == True):
            continue

        # Count frequency
        count = 1
        for j in range(i + 1, n, 1):
            if (arr[i] == arr[j]):
                visited[j] = True
                count += 1
        freq_list.append(count)
        print(arr[i], count)
    return freq_list
freq_list = countFreq(labels_array, len(labels_array))
balance_var = np.var(freq_list)/1000000

learning_rate = 'grid_search'
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
# with open("Kcache_list", "wb") as fp:
#     pickle.dump(Kcache_list, fp)
# with open("Kserver_list", "wb") as fp:
#     pickle.dump(Kserver_list, fp)
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
experiment(M,K=5,R=60,plotfile= plot_prefix + '/smallM-smallK-asynch-' + distribution + dataset_name + str(agg_scheme) + '.png',savearray=plot_prefix + '/smallM-smallK-asynch-' + distribution + dataset_name + str(agg_scheme))
#experiment(M=500,K=5,R=200,plotfile= plot_prefix + '/bigM-smallK-' + distribution + '.png',savearray=plot_prefix + '/bigM-smallK-' + distribution)
#experiment(M=50,K=40,R=200,plotfile= plot_prefix + '/smallM-bigK-' + distribution + '.png',savearray=plot_prefix + '/smallM-bigK-' + distribution)
#experiment(M=500,K=40,R=100,plotfile='plots/bigM-bigK-het.png',savearray='plots/bigM-bigK-het')
#experiment(M=50,K=200,R=100,plotfile='plots/smallM-biggerK-het.png',savearray='plots/smallM-biggerK-het')
