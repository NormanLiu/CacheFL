import sys
import numpy as np
import matplotlib
from sklearn.datasets import load_svmlight_file
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import os

np.set_printoptions(precision=3, linewidth=240, suppress=True)
np.random.seed(1993)

############################################## Logistic Regression ###############################################

def sigmoid(z):
    return 1. / (1. + np.exp(-np.clip(z, -15, 15)))

# features is an [n x d] matrix of features (each row is one data point)
# labels is an n-dimensional vector of labels (0/1)
def logistic_loss(x, features, labels):
    n = features.shape[0]
    probs = sigmoid(np.dot(features,x))
    return (-1./n) * (np.dot(labels, np.log(1e-12 + probs)) + np.dot(1-labels, np.log(1e-12 + 1-probs)))

def logistic_loss_full_gradient(x, features, labels):
    return np.dot(np.transpose(features), sigmoid(np.dot(features,x)) - labels) / features.shape[0]

def logistic_loss_stochastic_gradient(x, features, labels, minibatch_size):
    idxs = np.random.randint(0,features.shape[0],minibatch_size)
    fts = features[idxs,:]
    res = sigmoid(np.dot(fts,x)) - labels[idxs]
    return np.dot(res.reshape(1,minibatch_size), fts).reshape(len(x))/minibatch_size

def fedprox_loss_stochastic_gradient(x, features, labels, minibatch_size, xs):
    idxs = np.random.randint(0,features.shape[0],minibatch_size)
    fts = features[idxs,:]
    res = sigmoid(np.dot(fts,x)) - labels[idxs]
    return np.dot(res.reshape(1,minibatch_size), fts).reshape(len(x))/minibatch_size + 0.1*(x-xs)

def logistic_loss_hessian(x, features, labels):
    s = sigmoid(np.dot(features, x))
    s = s * (1 - s)
    return np.dot(np.transpose(features) * s, features) / features.shape[0]

##################################################################################################################

def one_inner_outer_iteration(x_start, M, K, stepsize, datasize):
    grads = np.zeros_like(x_start)
    for m in range(M):
        x = x_start.copy()
        for _ in range(K):
            #g = objective_stochastic_gradient(x, 1)
            g = logistic_loss_stochastic_gradient(x, features[m*datasize:(m+1)*datasize,:], labels[m*datasize:(m+1)*datasize], 1)
            grads += g / M
            x -= stepsize * g
    return grads

def one_inner_outer_iteration2(x_start_latest, x_start_delayed, M, K, stepsize, datasize):
    grads = np.zeros_like(x_start_latest)
    for m in range(M//2):
        x = x_start_latest.copy()
        for _ in range(K):
            #g = objective_stochastic_gradient(x, 1)
            g = logistic_loss_stochastic_gradient(x, features[m*datasize:(m+1)*datasize,:], labels[m*datasize:(m+1)*datasize], 1)
            grads += g / M
            x -= stepsize * g
    for m in range(M//2):
        x = x_start_delayed.copy()
        m += M//2
        for _ in range(K):
            #g = objective_stochastic_gradient(x, 1)
            g = logistic_loss_stochastic_gradient(x, features[m*datasize:(m+1)*datasize,:], labels[m*datasize:(m+1)*datasize], 1)
            grads += g / M
            x -= stepsize * g
#    grads

    return grads

def one_inner_outer_iteration3(x_start, M, K, stepsize, datasize):
    grads = np.zeros_like(x_start)
    """
    for m in range(M):
        x = x_start.copy()
        for _ in range(K):
            #g = objective_stochastic_gradient(x, 1)
            g = fedprox_loss_stochastic_gradient(x, features[m*datasize:(m+1)*datasize,:], labels[m*datasize:(m+1)*datasize], 1, x_start)
            grads += g / M
            x -= stepsize * g

    """
    for m in range(M // 2):
        x = x_start.copy()
        for _ in range(int(K)):
            g = fedprox_loss_stochastic_gradient(x, features[m*datasize:(m+1)*datasize,:], labels[m*datasize:(m+1)*datasize], 1, x_start)
            grads += g / M
            x -= stepsize * g
    for m in range(M // 2):
        x = x_start.copy()
        m += M // 2
        for _ in range(K):
            g = fedprox_loss_stochastic_gradient(x, features[m*datasize:(m+1)*datasize,:], labels[m*datasize:(m+1)*datasize], 1, x_start)
            grads += g / M
            x -= stepsize * g

    return grads

def inner_outer_sgd(x0_len, M, K, R, inner_stepsize, outer_stepsize, loss_freq, datasize, avg_window=8):
    losses = []
    iterates = [np.zeros(x0_len)]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        direction = one_inner_outer_iteration(iterates[-1], M, K, inner_stepsize, datasize)
        iterates.append(iterates[-1] - outer_stepsize * direction)
        if (r+1) % loss_freq == 0:
            losses.append(objective_value(np.average(iterates,axis=0)))
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
    print('')
    return losses

def inner_outer_sgd2(x0_len, M, K, R, inner_stepsize, outer_stepsize, loss_freq, datasize, avg_window=8):
    losses = []
    iterates = [np.zeros(x0_len)]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        if r>1:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-2], M, K, inner_stepsize, datasize)
        else:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-1], M, K, inner_stepsize, datasize)
        iterates.append(iterates[-1] - outer_stepsize * direction)
        if (r+1) % loss_freq == 0:
            losses.append(objective_value(np.average(iterates,axis=0)))
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
    print('')
    return losses



def inner_outer_sgd3(x0_len, M, K, R, inner_stepsize, outer_stepsize, loss_freq, datasize, avg_window=8):
    losses = []
    iterates = [np.zeros(x0_len)]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        direction = one_inner_outer_iteration3(iterates[-1], M, K, inner_stepsize, datasize)
        iterates.append(iterates[-1] - outer_stepsize * direction)
        if (r+1) % loss_freq == 0:
            model_accoracy(features, labels, iterates[-1])
            losses.append(objective_value(np.average(iterates,axis=0)))
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1,R,losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
    print('')
    return losses

def local_sgd(x0_len, M, K, R, stepsize, loss_freq, datasize):
    return inner_outer_sgd(x0_len, M, K, R, stepsize, stepsize, loss_freq, datasize)

def local_sgd_delayed(x0_len, M, K, R, stepsize, loss_freq, datasize):
    return inner_outer_sgd2(x0_len, M, K, R, stepsize, stepsize, loss_freq, datasize)

def fedprox(x0_len, M, K, R, stepsize, loss_freq, datasize):
    return inner_outer_sgd3(x0_len, M, K, R, stepsize, stepsize, loss_freq, datasize)

def minibatch_sgd(x0_len, T, batchsize, stepsize, loss_freq, avg_window=8):
    losses = []
    iterates = [np.zeros(x0_len)]
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

def gradient_descent(x0_len, T, stepsize):
    x = np.zeros(x0_len)
    losses = [objective_value(x)]
    for t in range(T):
        x -= stepsize * objective_full_gradient(x)
        losses.append(objective_value(x))
    return np.array(losses)


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

def model_accoracy(X_test, y_test, w):
    #X_test = np.insert(X_test, 0, 1, axis=1)  # add constant
    predictions = []  # define a prediction list
    for i in range(X_test.shape[0]):  # iterate n samples
        prob = sigmoid(np.dot(X_test[i], w))  # softmax
        predict = round(prob)  # find the index with maximum probability
        predictions.append(predict)  # add the final prediction to the list

    accuracy = np.count_nonzero(pd.Series(predictions) == pd.Series(y_test)) / len(predictions)
    print("Our test accuracy is ", accuracy)

##################################################################################################################

def experiment(M,K,R,plotfile,savearray):
    loss_freq = 5
    n_reps = 5
    datasize = N//M

    print('Doing FedProx...')
    prox_results = np.zeros((R // loss_freq, len(lc_stepsizes)))
    for i, stepsize in enumerate(lc_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(lc_stepsizes)))
        for rep in range(n_reps):
            prox_results[:, i] += (fedprox(x0_len, M, K, R, stepsize, loss_freq, datasize) - fstar) / n_reps

    print('Doing Partial Delayed Local SGD...')
    delay_results = np.zeros((R//loss_freq, len(tt_stepsizes)))
    for i,stepsize in enumerate(tt_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(tt_stepsizes)))
        for rep in range(n_reps):
            try:
                delay_results[:, i] += (local_sgd_delayed(x0_len, M, K, R, stepsize, loss_freq,
                                                          datasize) - fstar) / n_reps
            except ValueError:
                delay_results[:, i]
                (local_sgd_delayed(x0_len, M, K, R, stepsize, loss_freq, datasize) - fstar) / n_reps
                print('Catch a ValueError')
                break

    print('Doing Local SGD with less workers...')
    less_results = np.zeros((R//loss_freq, len(lg_stepsizes)))
    for i,stepsize in enumerate(lg_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(lg_stepsizes)))
        for rep in range(n_reps):
            less_results[:,i] += (local_sgd(x0_len, M//2, K, R, stepsize, loss_freq, datasize) - fstar) / n_reps

    print('Doing Local SGD...')
    local_results = np.zeros((R//loss_freq, len(lc_stepsizes)))
    for i,stepsize in enumerate(lc_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(lc_stepsizes)))
        for rep in range(n_reps):
            local_results[:,i] += (local_sgd(x0_len, M, K, R, stepsize, loss_freq, datasize) - fstar) / n_reps



    l0 = objective_value(np.zeros(x0_len))-fstar
    local_l = np.concatenate([[l0], np.min(local_results, axis=1)])
    delay_l = np.concatenate([[l0], np.min(delay_results, axis=1)])
    less_l = np.concatenate([[l0], np.min(less_results, axis=1)])
    prox_l = np.concatenate([[l0], np.min(prox_results, axis=1)])

    np.savez(savearray, local_l, delay_l, less_l, prox_l)

    Rs = [0] + list(range(loss_freq, R+1, loss_freq))

    # plot results
    #plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Rs, local_l, label='Local SGD')
    ax.plot(Rs, delay_l, label='Delayed Local SGD')
    ax.plot(Rs, less_l, label='Local SGD with less workers')
    ax.plot(Rs, prox_l, label='FedProx')
    handles,labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Round of Communication')
    ax.set_ylabel('Objective Value')
    ax.set_title('K={:d}, tau={:d}'.format(M, K))
    ax.legend(handles, labels, loc='upper right')
    #plt.ioff()
    plt.savefig(plotfile)


##################################################################################################################

frac_delay = sys.argv[1] if len(sys.argv) > 1 else 0.5
learning_rate = sys.argv[2] if len(sys.argv) > 2 else 0.01
distribution = sys.argv[3] if len(sys.argv) > 3 else 'het'
dataset = sys.argv[4] if len(sys.argv) > 4 else 'synthetic'
plot_prefix = sys.argv[5] if len(sys.argv) > 5 else 'plots'
dataset_name = sys.argv[6] if len(sys.argv) > 6 else 'a5a'
print([frac_delay, learning_rate, distribution, dataset, plot_prefix, dataset_name])

if dataset == 'synthetic':
    # number of samples on each device follows a power law
    M = 50
    n_sample = 100 * np.random.zipf(2, M)
    scale = 1
    dim = 25
    N = sum(n_sample)
    Sigma = [(j + 1) ** (-1.2) for j in range(dim)]
    if distribution == 'iid':
        vk = np.random.randn() * np.ones(M)
    else:
        vk = np.random.randn(M)
    for k in range(M):
        if k == 0:
            features = np.random.normal(vk[k], Sigma, (n_sample[k], dim))
        else:
            features_k = np.random.normal(np.random.randn(), Sigma, (n_sample[k], dim))
            features = np.concatenate([features, features_k], axis=0)
    #features = scale * np.dot(np.random.randn(N, dim), np.diag((np.linspace(1./dim, 1., dim))**2)) # [N x d]
    w1 = np.random.randn(dim)/(scale*np.sqrt(dim))
    w2 = np.random.randn(dim)/(scale*np.sqrt(dim))
    b1 = 4*np.random.randn(1)
    b2 = 4*np.random.randn(1)
    prob_positive = sigmoid(np.minimum(np.dot(features, w1)+b1, np.dot(features, w2)+b2))
    labels = np.random.binomial(1, prob_positive)
    features = np.concatenate([features, np.ones((N,1))], axis=1) # for bias term
    x0_len = features.shape[1]
elif dataset == 'real':
    data = load_svmlight_file(dataset_name)
    features, labels = data[0].toarray(), (data[1] + 1) // 2
    N = features.shape[0]
    features = np.concatenate([features, np.ones((N, 1))], axis=1)
    x0_len = features.shape[1]
    if distribution != 'iid':
        sample = np.concatenate([features, np.reshape(labels, (N, 1))], axis=1)
        sample = sample[sample[:, -1].argsort()]
        if distribution == 'het2':
            sample = np.flip(sample, axis=0)
        features = sample[:, 0:-1]
        labels = sample[:, -1]

if learning_rate == 'grid_search':
    n_stepsizes = 10
    tt_stepsizes = [np.exp(exponent) for exponent in np.linspace(-8, -1, n_stepsizes)]
    lg_stepsizes = [np.exp(exponent) for exponent in np.linspace(-10, -1, n_stepsizes)]
    lc_stepsizes = [np.exp(exponent) for exponent in np.linspace(-8, -1, n_stepsizes)]
else:
    tt_stepsizes = [learning_rate]
    lg_stepsizes = [learning_rate]
    lc_stepsizes = [learning_rate]

loss_function = 'binary logistic loss'
objective_value = lambda x: logistic_loss(x, features, labels) #+ 0.05*np.linalg.norm(x)**2
objective_full_gradient = lambda x: logistic_loss_full_gradient(x, features, labels) #+ 0.1*x
objective_stochastic_gradient = lambda x, minibatch_size: logistic_loss_stochastic_gradient(x, features, labels, minibatch_size) #+ 0.1*x
objective_hessian = lambda x: logistic_loss_hessian(x, features, labels) #+ 0.1*np.eye(len(x))
fedprox_stochastic_gradient = lambda x, xg, minibatch_size: logistic_loss_stochastic_gradient(x, features, labels, minibatch_size) + 0.1*(x-xg)

fstar = newtons_method(x0_len)
#fstar = np.float64(0)
print('Fstar = {:.5f}'.format(fstar))

experiment(M=10,K=5,R=200,plotfile= plot_prefix + '/smallerM-smallK-' + distribution + '.png',savearray=plot_prefix + '/smallerM-smallK-' + distribution)
experiment(M=50,K=5,R=200,plotfile= plot_prefix + '/smallM-smallK-' + distribution + '.png',savearray=plot_prefix + '/smallM-smallK-' + distribution)
experiment(M=500,K=5,R=200,plotfile= plot_prefix + '/bigM-smallK-' + distribution + '.png',savearray=plot_prefix + '/bigM-smallK-' + distribution)
experiment(M=50,K=40,R=200,plotfile= plot_prefix + '/smallM-bigK-' + distribution + '.png',savearray=plot_prefix + '/smallM-bigK-' + distribution)
#experiment(M=500,K=40,R=100,plotfile='plots/bigM-bigK-het.png',savearray='plots/bigM-bigK-het')
#experiment(M=50,K=200,R=100,plotfile='plots/smallM-biggerK-het.png',savearray='plots/smallM-biggerK-het')
