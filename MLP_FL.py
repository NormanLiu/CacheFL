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
from scipy.special import expit as activation_function  # 1/(1+exp(-x)), sigmoid
from scipy.stats import truncnorm
#from __future__ import division

np.set_printoptions(precision=3, linewidth=240, suppress=True)
random.seed(1993)
np.random.seed(2023)

############################################## MLP ###############################################
def create_weight_matrices(structure):
    bias_node = 0
    no_of_layers = len(structure)
    weights_matrices = []
    for k in range(no_of_layers - 1):
        nodes_in = structure[k]
        nodes_out = structure[k + 1]
        n = (nodes_in + bias_node) * nodes_out
        X = truncnorm(-1, 1, loc=0, scale=1)
        wm = X.rvs(n).reshape((nodes_out, nodes_in + bias_node))
        weights_matrices.append(wm)
    return weights_matrices

def train(weights_matrices, input_vector, target_vector, learning_rate):
    input_vector = np.array(input_vector, ndmin=2).T
    res_vectors = [input_vector]
    no_of_layers = len(weights_matrices) + 1
    grads = []
    for k in range(no_of_layers-1):
        in_vector = res_vectors[-1]
        x = np.dot(weights_matrices[k], in_vector)
        out_vector = activation_function(x)
        res_vectors.append(out_vector)

    target_vector = np.array(target_vector, ndmin=2).T
    output_errors = target_vector - out_vector
    for k in range(no_of_layers-1, 0, -1):
        out_vector = res_vectors[k]
        in_vector = res_vectors[k-1]
        tmp = output_errors * out_vector * (1.0 - out_vector)  # sigma'(x) = sigma(x) (1 - sigma(x))
        tmp = np.dot(tmp, in_vector.T)
        grads.append(tmp)
        weights_matrices[k-1] += learning_rate * tmp
        output_errors = np.dot(weights_matrices[k-1].T, output_errors)
    grads.reverse()
    return grads

def MLP_stochastic_gradient(x, features, labels, minibatch_size, learning_rate):
    #grads = np.zeros_like(x)
    grads = [np.zeros_like(x[i]) for i in range(len(structure) - 1)]
    idxs = np.random.randint(0, features.shape[0], minibatch_size)
    fts = features[idxs, :]
    lts = labels[idxs, :]
    for i in range(minibatch_size):
        tmp = train(x, fts[i], lts[i], learning_rate)
        grads = [(grads[j] + tmp[j])/minibatch_size for j in range(len(structure)-1)]
    return grads

def fedprox_MLP_stochastic_gradient(x, features, labels, minibatch_size, learning_rate, xs):
    #grads = np.zeros_like(x)
    grads = [np.zeros_like(x[i]) for i in range(len(structure) - 1)]
    idxs = np.random.randint(0, features.shape[0], minibatch_size)
    fts = features[idxs, :]
    lts = labels[idxs, :]
    for i in range(minibatch_size):
        tmp = train(x, fts[i], lts[i], learning_rate)
        grads = [(grads[j] + tmp[j])/minibatch_size for j in range(len(structure)-1)]
    grads = [(grads[i] + 0.1*x[i] - 0.1*xs[i]) for i in range(len(structure) - 1)]
    return grads

def run(weights_matrices, input_vector):
    in_vector = np.array(input_vector, ndmin=2).T
    for k in range(len(structure)-1):
        x = np.dot(weights_matrices[k], in_vector)
        out_vector = activation_function(x)
        in_vector = out_vector
    return out_vector

def evaluate(weights_matrices, data, labels):
    corrects, wrongs= 0, 0
    for i in range(len(data)):
        res = run(weights_matrices, data[i])
        res_max = res.argmax()
        if res_max == labels[i]:
            corrects += 1
        else:
            wrongs += 1
    return corrects / (corrects + wrongs)

def MLP_loss(weights_matrices, data, labels_one_hot):
    loss = 0
    n = features.shape[0]
    for i in range(len(data)):
        res = run(weights_matrices, data[i])
        error_vector = np.array(labels_one_hot[i], ndmin=2).T - res
        loss += np.dot(np.transpose(error_vector), error_vector)
    return (1./n)*float(loss)/2

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
    #grads = np.zeros_like(x_start)
    grads = [np.zeros_like(x_start[i]) for i in range(len(structure)-1)]
    for m in Kset:
        x = x_start.copy()
        for _ in range(K):
            g = MLP_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 5, stepsize)
            #grads += g * n_sample_fraction[m]
            grads = [(grads[i]+g[i]*n_sample_fraction[m]) for i in range(len(structure)-1)]
            #x -= stepsize * g
            x = [(x[i] - stepsize*g[i]) for i in range(len(structure)-1)]
    return grads

def one_inner_outer_iteration2(x_start_latest, x_start_delayed, Kcache, Kserver, K, stepsize):
    grads = [np.zeros_like(x_start_latest[i]) for i in range(len(structure)-1)]
    for m in Kserver:
        x = x_start_latest.copy()
        for _ in range(K):
            g = MLP_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 5, stepsize)
            # grads += g * n_sample_fraction[m]
            grads = [(grads[i] + g[i] * n_sample_fraction[m]) for i in range(len(structure) - 1)]
            # x -= stepsize * g
            x = [(x[i] - stepsize * g[i]) for i in range(len(structure) - 1)]
    for m in Kcache:
        x = x_start_delayed.copy()
        for _ in range(K):
            g = MLP_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 5, stepsize)
            # grads += g * n_sample_fraction[m]
            grads = [(grads[i] + g[i] * n_sample_fraction[m]) for i in range(len(structure) - 1)]
            # x -= stepsize * g
            x = [(x[i] - stepsize * g[i]) for i in range(len(structure) - 1)]
#    grads
    return grads

def one_inner_outer_iteration3(x_start, M, K, stepsize):
    grads = [np.zeros_like(x_start[i]) for i in range(len(structure)-1)]
    for m in range(M):
        x = x_start.copy()
        for _ in range(K):
            g = fedprox_MLP_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels[range_samples[m]:range_samples[m+1]], 5, stepsize, x_start)
            # grads += g * n_sample_fraction[m]
            grads = [(grads[i] + g[i] * n_sample_fraction[m]) for i in range(len(structure) - 1)]
            # x -= stepsize * g
            x = [(x[i] - stepsize * g[i]) for i in range(len(structure) - 1)]
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
    #iterates = [np.zeros((x0_len, C))]
    iterates = [create_weight_matrices(structure)]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        direction = one_inner_outer_iteration(iterates[-1], Kset, K, inner_stepsize)
        #iterates.append(iterates[-1] - outer_stepsize * direction)
        current_iterate = iterates[-1]
        iterates.append([(current_iterate[i] - outer_stepsize * direction[i]) for i in range(len(structure)-1)])
        if (r+1) % loss_freq == 0:
            losses.append(MLP_loss(iterates[-1], features, labels))
            accuracies.append(evaluate(iterates[-1], features_t, labels_t))
            print('Iteration: {:d}/{:d}   Accuracy: {:f}                 \r'.format(r + 1, R, accuracies[-1]), end='')
    print('')
    return [losses, accuracies]

def inner_outer_sgd2(x0_len, Kcache, Kserver, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
    losses = []
    accuracies = []
    # iterates = [np.zeros((x0_len, C))]
    iterates = [create_weight_matrices(structure)]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        if r>1:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-2], Kcache, Kserver, K, inner_stepsize)
        else:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-1], Kcache, Kserver, K, inner_stepsize)
        current_iterate = iterates[-1]
        iterates.append([(current_iterate[i] - outer_stepsize * direction[i]) for i in range(len(structure) - 1)])
        if (r+1) % loss_freq == 0:
            losses.append(MLP_loss(iterates[-1], features, labels))
            accuracies.append(evaluate(iterates[-1], features_t, labels_t))
            print('Iteration: {:d}/{:d}   Accuracy: {:f}                 \r'.format(r + 1, R, accuracies[-1]), end='')
    print('')
    return [losses, accuracies]



def inner_outer_sgd3(x0_len, M, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
    losses = []
    accuracies = []
    # iterates = [np.zeros((x0_len, C))]
    iterates = [create_weight_matrices(structure)]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        direction = one_inner_outer_iteration3(iterates[-1], M, K, inner_stepsize)
        current_iterate = iterates[-1]
        iterates.append([(current_iterate[i] - outer_stepsize * direction[i]) for i in range(len(structure) - 1)])
        if (r+1) % loss_freq == 0:
            losses.append(MLP_loss(iterates[-1], features, labels))
            accuracies.append(evaluate(iterates[-1], features_t, labels_t))
            print('Iteration: {:d}/{:d}   Accuracy: {:f}                 \r'.format(r + 1, R, accuracies[-1]), end='')
    print('')
    return [losses, accuracies]

def inner_outer_sgd4(x0_len, M, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
    losses = []
    accuracies = []
    # iterates = [np.zeros((x0_len, C))]
    iterates = [create_weight_matrices(structure)]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        if r>1:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-2], Kcache_list[r//10], Kserver_list[r//10], K, inner_stepsize)
        else:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-1], Kcache_list[r//10], Kserver_list[r//10], K, inner_stepsize)
        current_iterate = iterates[-1]
        iterates.append([(current_iterate[i] - outer_stepsize * direction[i]) for i in range(len(structure) - 1)])
        if (r+1) % loss_freq == 0:
            losses.append(MLP_loss(iterates[-1], features, labels))
            accuracies.append(evaluate(iterates[-1], features_t, labels_t))
            print('Iteration: {:d}/{:d}   Accuracy: {:f}                 \r'.format(r + 1, R, accuracies[-1]), end='')
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
    loss_freq = 1
    n_reps = 2

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

    print('Doing Random Delayed Local SGD...')
    random_results = np.zeros((R // loss_freq, len(tt_stepsizes)))
    random_results_acc = np.zeros((R // loss_freq, len(tt_stepsizes)))
    for i, stepsize in enumerate(tt_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(tt_stepsizes)))
        for rep in range(n_reps):
            [losses, accuracies] = local_sgd_delayed_random(x0_len, M, K, R, stepsize, loss_freq)
            random_results[:, i] += (losses - fstar) / n_reps
            random_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps

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
    local_results = np.zeros((R // loss_freq, len(lc_stepsizes)))
    local_results_acc = np.zeros((R // loss_freq, len(lc_stepsizes)))
    for i, stepsize in enumerate(lc_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(lc_stepsizes)))
        for rep in range(n_reps):
            [losses, accuracies] = local_sgd(x0_len, range(M), K, R, stepsize, loss_freq)
            local_results[:, i] += (losses - fstar) / n_reps
            local_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps

    x0 = create_weight_matrices(structure)
    l0 = MLP_loss(x0, features, labels) - fstar
    local_l = np.concatenate([[l0], np.min(local_results, axis=1)])
    delay_l = np.concatenate([[l0], np.min(delay_results, axis=1)])
    less_l = np.concatenate([[l0], np.min(less_results, axis=1)])
    prox_l = np.concatenate([[l0], np.min(prox_results, axis=1)])
    random_l = np.concatenate([[l0], np.min(random_results, axis=1)])

    a0 = evaluate(x0, features_t, labels_t)
    local_a = np.concatenate([[a0], np.max(local_results_acc, axis=1)])
    delay_a = np.concatenate([[a0], np.max(delay_results_acc, axis=1)])
    less_a = np.concatenate([[a0], np.max(less_results_acc, axis=1)])
    prox_a = np.concatenate([[a0], np.max(prox_results_acc, axis=1)])
    random_a = np.concatenate([[a0], np.max(random_results_acc, axis=1)])

    np.savez(savearray + 'grid_search', local_results=local_results, local_results_acc=local_results_acc, less_results=less_results,less_results_acc=less_results_acc,
             prox_results=prox_results, prox_results_acc=prox_results_acc, random_results=random_results, random_results_acc=random_results_acc, delay_results=delay_results, delay_results_acc=delay_results_acc)
    np.savez(savearray, local_l=local_l, delay_l=delay_l, less_l=less_l, prox_l=prox_l, random_l=random_l,
             local_a=local_a, delay_a=delay_a, less_a=less_a, prox_a=prox_a, random_a=random_a)

    Rs = [0] + list(range(loss_freq, R+1, loss_freq))

    # generate delay for each case
    TcacheFL = list(np.arange(0, (R + 1) * Tsingle_cacheFL, loss_freq * Tsingle_cacheFL))
    Tregular = list(np.arange(0, (R + 1) * Tsingle_regular, loss_freq * Tsingle_regular))
    Tless = list(np.arange(0, (R + 1) * Tsingle_less, loss_freq * Tsingle_less))
    Trandom = [0]
    current_T = 0
    for r in range(R // loss_freq):
        current_T += loss_freq * Tsingle_random[r % 20]
        Trandom.append(current_T)

    Tregular = [x for x in Tregular if x <= TcacheFL[-1]]
    Trandom = [x for x in Trandom if x <= TcacheFL[-1]]

    # plot results
    fig = plt.figure(figsize=(12.8, 9.6))
    ax = fig.add_subplot(221)
    ax.plot(Rs, local_a, "v-", label='FedAvg')
    ax.plot(Rs, delay_a, "o-", label='CacheFL-OPT')
    ax.plot(Rs, less_a, "s-", label='FedAvg stragglers')
    ax.plot(Rs, prox_a, "x-", label='FedProx')
    ax.plot(Rs, random_a, "^-", label='CacheFL-random')
    handles,labels_fig = ax.get_legend_handles_labels()
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Test Accuracy')
    #ax.set_title('K={:d}, tau={:d}'.format(M, K))
    ax.legend(handles, labels_fig, loc='lower right')


    ax = fig.add_subplot(222)
    ax.plot(Tregular, local_a[0:len(Tregular)], "v-", label='FedAvg')
    ax.plot(TcacheFL, delay_a[0:len(TcacheFL)], "o-", label='CacheFL-OPT')
    ax.plot(Tless, less_a[0:len(Tless)], "s-", label='FedAvg stragglers')
    ax.plot(Tregular, prox_a[0:len(Tregular)], "x-", label='FedProx')
    ax.plot(Trandom, random_a[0:len(Trandom)], "^-", label='CacheFL-random')
    handles, labels_fig = ax.get_legend_handles_labels()
    ax.set_xlabel('Wall-clock Time')
    ax.set_ylabel('Test Accuracy')
    # ax.set_title('K={:d}, tau={:d}'.format(M, K))
    ax.legend(handles, labels_fig, loc='lower right')

    ax = fig.add_subplot(223)
    ax.plot(Rs, local_l, "v-", label='FedAvg')
    ax.plot(Rs, delay_l, "o-", label='CacheFL-OPT')
    ax.plot(Rs, less_l, "s-", label='FedAvg stragglers')
    ax.plot(Rs, prox_l, "x-", label='FedProx')
    ax.plot(Rs, random_l, "^-", label='CacheFL-random')
    handles, labels_fig = ax.get_legend_handles_labels()
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Training Loss')
    # ax.set_title('K={:d}, tau={:d}'.format(M, K))
    ax.legend(handles, labels_fig, loc='lower right')

    ax = fig.add_subplot(224)
    ax.plot(Tregular, local_l[0:len(Tregular)], "v-", label='FedAvg')
    ax.plot(TcacheFL, delay_l[0:len(TcacheFL)], "o-", label='CacheFL-OPT')
    ax.plot(Tless, less_l[0:len(Tless)], "s-", label='FedAvg stragglers')
    ax.plot(Tregular, prox_l[0:len(Tregular)], "x-", label='FedProx')
    ax.plot(Trandom, random_l[0:len(Trandom)], "^-", label='CacheFL-random')
    handles, labels_fig = ax.get_legend_handles_labels()
    ax.set_xlabel('Wall-clock Time')
    ax.set_ylabel('Training Loss')
    # ax.set_title('K={:d}, tau={:d}'.format(M, K))
    ax.legend(handles, labels_fig, loc='lower right')

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
dataset_name = sys.argv[6] if len(sys.argv) > 6 else 'cifar10'
print([frac_delay, learning_rate, distribution, dataset, plot_prefix, dataset_name])



#generate data
C = 10 # number of classes
# number of samples on each device follows a power law
M = 50
# n_sample = 100*np.random.zipf(2, M)
# np.savez('n_sample_more', n_sample = n_sample)
#npzfile = np.load('n_sample.npz')
if dataset_name == 'mnist.scale':
    npzfile = np.load('n_sample_balance_less.npz')
elif dataset_name == 'cifar10':
    npzfile = np.load('n_sample_mnist.npz')
n_sample = npzfile['n_sample']
n_sample = n_sample//5
range_samples = [sum(n_sample[0:i]) for i in range(len(n_sample)+1)]
n_sample_fraction = n_sample/sum(n_sample)

if dataset_name == 'mnist.scale':
    npzfile = np.load('random_delay_balance_less.npz')
elif dataset_name == 'cifar10':
    npzfile = np.load('random_delay_traces_asynch_mnist.npz')
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


data = load_svmlight_file(dataset_name)
features, labels_array = data[0].toarray(), data[1]
if dataset_name == 'cifar10':
    features = features/255
features_t, labels_t = features, labels_array
if distribution == 'iid':
    features = features[0:range_samples[-1]]
    labels_array = labels_array[0:range_samples[-1]]
elif distribution == 'het':
    samples_idx = []
    for k in range(M):
        if dataset_name == 'mnist.scale':
            classes = np.random.choice(10, 1, replace=False)
            candidate1 = np.where(labels_array == classes[0])
            samples_idx += np.ndarray.tolist(np.random.choice(candidate1[0], n_sample[k], replace=True))
        else:
            classes = np.random.choice(10, 2, replace=False)
            candidate1 = np.where(labels_array == classes[0])
            candidate2 = np.where(labels_array == classes[1])
            candidate = np.concatenate([candidate1[0], candidate2[0]])
            samples_idx += np.ndarray.tolist(np.random.choice(candidate, n_sample[k], replace=True))
        # classes = np.random.choice(10, 2, replace=False)
        # candidate1 = np.where(labels_array == classes[0])
        # candidate2 = np.where(labels_array == classes[1])
        # candidate = np.concatenate([candidate1[0], candidate2[0]])
        # samples_idx += np.ndarray.tolist(np.random.choice(candidate, n_sample[k], replace=False))
    features = features[samples_idx]
    labels_array = labels_array[samples_idx]
N = features.shape[0]
labels = np.zeros((N, C))
for i in range(N):
    labels[i, int(labels_array[i])] = 1

if dataset_name == 'mnist.scale':
    structure = [780, 200, 80, 10]
elif dataset_name == 'cifar10':
    structure = [3072, 256, 256, 10]

# generate stepsizes
if learning_rate == 'grid_search':
    n_stepsizes = 4
    tt_stepsizes = [np.exp(exponent) for exponent in np.linspace(-5, -3, n_stepsizes)]
    lg_stepsizes = [np.exp(exponent) for exponent in np.linspace(-5, -3, n_stepsizes)]
    lc_stepsizes = [np.exp(exponent) for exponent in np.linspace(-5, -3, n_stepsizes)]
else:
    tt_stepsizes = [float(learning_rate)]
    lg_stepsizes = [float(learning_rate)]
    lc_stepsizes = [float(learning_rate)]


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
experiment(M,K=3,R=60,plotfile= plot_prefix + '/smallM-smallK-' + distribution + dataset_name + '-MLP.png',savearray=plot_prefix + '/smallM-smallK-' + distribution + dataset_name+'-MLP')
#experiment(M=500,K=5,R=200,plotfile= plot_prefix + '/bigM-smallK-' + distribution + '.png',savearray=plot_prefix + '/bigM-smallK-' + distribution)
#experiment(M=50,K=40,R=200,plotfile= plot_prefix + '/smallM-bigK-' + distribution + '.png',savearray=plot_prefix + '/smallM-bigK-' + distribution)
