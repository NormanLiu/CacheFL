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
import gzip
from scipy.special import expit as activation_function  # 1/(1+exp(-x)), sigmoid
from scipy.stats import truncnorm
#from __future__ import division

np.set_printoptions(precision=3, linewidth=240, suppress=True)
random.seed(1993)
np.random.seed(2023)


#####################################################
################ Forward Operations #################
#####################################################

def convolution(image, filt, bias, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape  # filter dimensions
    n_c, in_dim, _ = image.shape  # image dimensions

    out_dim = int((in_dim - f) / s) + 1  # calculate output dimensions

    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"

    out = np.zeros((n_f, out_dim, out_dim))

    # convolve the filter over every part of the image, adding the bias at each step.
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + f, curr_x:curr_x + f]) + \
                                            bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return out


def maxpool(image, f=2, s=2):
    '''
    Downsample `image` using kernel size `f` and stride `s`
    '''
    n_c, h_prev, w_prev = image.shape

    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    downsampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        # slide maxpool window over each part of the image and assign the max value at each step to the output
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y + f, curr_x:curr_x + f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled


def softmax(X):
    out = np.exp(np.clip(X, -15, 15))
    return out / np.sum(out)


def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))


#####################################################
############### Backward Operations #################
#####################################################

def convolutionBackward(dconv_prev, conv_in, filt, s):
    '''
    Backpropagation through a convolutional layer.
    '''
    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    ## initialize derivatives
    dout = np.zeros(conv_in.shape)
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f, 1))
    for curr_f in range(n_f):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # loss gradient of filter (used to update the filter)
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y + f, curr_x:curr_x + f]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[:, curr_y:curr_y + f, curr_x:curr_x + f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # loss gradient of the bias
        dbias[curr_f] = np.sum(dconv_prev[curr_f])

    return dout, dfilt, dbias


def maxpoolBackward(dpool, orig, f, s):
    '''
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
    '''
    (n_c, orig_dim, _) = orig.shape

    dout = np.zeros(orig.shape)

    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
                dout[curr_c, curr_y + a, curr_x + b] = dpool[curr_c, out_y, out_x]

                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return dout

#####################################################
############### Building The Network ################
#####################################################

def conv(image, label, params, conv_s=1, pool_f=2, pool_s=2):
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    image = image.reshape(1, 28, 28)
    label = np.eye(10)[int(label)].reshape(10, 1)

    ################################################
    ############## Forward Operation ###############
    ################################################
    conv1 = convolution(image, f1, b1, conv_s)  # convolution operation
    conv1[conv1 <= 0] = 0  # pass through ReLU non-linearity

    conv2 = convolution(conv1, f2, b2, conv_s)  # second convolution operation
    conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

    pooled = maxpool(conv2, pool_f, pool_s)  # maxpooling operation

    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer

    z = w3.dot(fc) + b3  # first dense layer
    z[z <= 0] = 0  # pass through ReLU non-linearity

    out = w4.dot(z) + b4  # second dense layer

    probs = softmax(out)  # predict class probabilities with the softmax activation function

    ################################################
    #################### Loss ######################
    ################################################

    loss = categoricalCrossEntropy(probs, label)  # categorical cross-entropy loss

    ################################################
    ############# Backward Operation ###############
    ################################################
    dout = probs - label  # derivative of loss w.r.t. final dense layer output
    dw4 = dout.dot(z.T)  # loss gradient of final dense layer weights
    db4 = np.sum(dout, axis=1).reshape(b4.shape)  # loss gradient of final dense layer biases

    dz = w4.T.dot(dout)  # loss gradient of first dense layer outputs
    dz[z <= 0] = 0  # backpropagate through ReLU
    dw3 = dz.dot(fc.T)
    db3 = np.sum(dz, axis=1).reshape(b3.shape)

    dfc = w3.T.dot(dz)  # loss gradients of fully-connected layer (pooling layer)
    dpool = dfc.reshape(pooled.shape)  # reshape fully connected into dimensions of pooling layer

    dconv2 = maxpoolBackward(dpool, conv2, pool_f,
                             pool_s)  # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
    dconv2[conv2 <= 0] = 0  # backpropagate through ReLU

    dconv1, df2, db2 = convolutionBackward(dconv2, conv1, f2,
                                           conv_s)  # backpropagate previous gradient through second convolutional layer.
    dconv1[conv1 <= 0] = 0  # backpropagate through ReLU

    dimage, df1, db1 = convolutionBackward(dconv1, image, f1,
                                           conv_s)  # backpropagate previous gradient through first convolutional layer.

    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

    return grads, loss


#####################################################
################## Utility Methods ##################
#####################################################

def extract_data(filename, num_images, IMAGE_WIDTH):
    '''
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w], where m
    is the number of training examples.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH * IMAGE_WIDTH)
        return data


def extract_labels(filename, num_images):
    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def initializeFilter(size, scale=1.0):
    stddev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)


def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01


def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs

############################################## CNN ###############################################
def create_weight_matrices(img_depth = 1, f = 5, num_filt1 = 8, num_filt2 = 8):
    ## Initializing all the parameters
    f1, f2, w3, w4 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f), (128, 800), (10, 128)
    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    w3 = initializeWeight(w3)
    w4 = initializeWeight(w4)

    b1 = np.zeros((f1.shape[0], 1))
    b2 = np.zeros((f2.shape[0], 1))
    b3 = np.zeros((w3.shape[0], 1))
    b4 = np.zeros((w4.shape[0], 1))

    params = [f1, f2, w3, w4, b1, b2, b3, b4]
    return params


def MLP_stochastic_gradient(x, features, labels, minibatch_size):
    grads = [np.zeros_like(x[i]) for i in range(len(x))]
    idxs = np.random.randint(0, features.shape[0], minibatch_size)
    fts = features[idxs, :]
    lts = labels[idxs]
    #fts = fts.reshape(minibatch_size, 1, 28, 28)
    for i in range(minibatch_size):
        tmp, _ = conv(fts[i], lts[i], x)
        grads = [(grads[j] + tmp[j])/minibatch_size for j in range(len(x))]
    return grads

def fedprox_MLP_stochastic_gradient(x, features, labels, minibatch_size, xs):
    grads = [np.zeros_like(x[i]) for i in range(len(x))]
    idxs = np.random.randint(0, features.shape[0], minibatch_size)
    fts = features[idxs, :]
    lts = labels[idxs]
    for i in range(minibatch_size):
        tmp, _ = conv(fts[i], lts[i], x)
        grads = [(grads[j] + tmp[j])/minibatch_size for j in range(len(x))]
    grads = [(grads[i] + 0.1*x[i] - 0.1*xs[i]) for i in range(len(x))]
    return grads

def run(weights_matrices, input_vector, conv_s=1, pool_f=2, pool_s=2):
    [f1, f2, w3, w4, b1, b2, b3, b4] = weights_matrices

    input_vector = input_vector.reshape(1, 28, 28)

    ################################################
    ############## Forward Operation ###############
    ################################################
    conv1 = convolution(input_vector, f1, b1, conv_s)  # convolution operation
    conv1[conv1 <= 0] = 0  # pass through ReLU non-linearity

    conv2 = convolution(conv1, f2, b2, conv_s)  # second convolution operation
    conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

    pooled = maxpool(conv2, pool_f, pool_s)  # maxpooling operation

    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer

    z = w3.dot(fc) + b3  # first dense layer
    z[z <= 0] = 0  # pass through ReLU non-linearity

    out = w4.dot(z) + b4  # second dense layer

    return out

def evaluate(weights_matrices, data, labels):
    corrects, wrongs= 0, 0
    loss = 0
    n = data.shape[0]
    for i in range(len(data)):
        res = softmax(run(weights_matrices, data[i]))
        res_max = res.argmax()
        if res_max == labels[i]:
            corrects += 1
        else:
            wrongs += 1
        y = np.eye(10)[int(labels[i])].reshape(10, 1)
        loss += categoricalCrossEntropy(res, y)
    return (1./n)*float(loss)/2, corrects / (corrects + wrongs)

# def MLP_loss(weights_matrices, data, labels_one_hot):
#     loss = 0
#     n = data.shape[0]
#     for i in range(len(data)):
#         res = softmax(run(weights_matrices, data[i]))
#         #error_vector = np.array(labels_one_hot[i], ndmin=2).T - res
#         #loss += np.dot(np.transpose(error_vector), error_vector)
#         loss += categoricalCrossEntropy(res, labels_one_hot[i])
#     return (1./n)*float(loss)/2

# def softmax(x):
#     probs = np.exp(np.clip(x, -15, 15))   # n x C or C-dim vector
#     if probs.ndim == 1:
#         divide = np.sum(probs) # 1
#         probs /= divide
#     else:
#         divide = np.sum(probs, axis=1)  # n x 1
#         for i in range(x.shape[0]):
#             probs[i] /= divide[i]
#     return probs


##################################################################################################################

def one_inner_outer_iteration(x_start, Kset, K, stepsize):
    #grads = np.zeros_like(x_start)
    grads = [np.zeros_like(x_start[i]) for i in range(len(x_start))]
    for m in Kset:
        x = x_start.copy()
        for _ in range(K):
            g = MLP_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels_array[range_samples[m]:range_samples[m+1]], 3)
            #grads += g * n_sample_fraction[m]
            grads = [(grads[i]+g[i]*n_sample_fraction[m]) for i in range(len(x_start))]
            #x -= stepsize * g
            x = [(x[i] - stepsize*g[i]) for i in range(len(x_start))]
    return grads

def one_inner_outer_iteration2(x_start_latest, x_start_delayed, Kcache, Kserver, K, stepsize):
    grads = [np.zeros_like(x_start_latest[i]) for i in range(len(x_start_latest))]
    for m in Kserver:
        x = x_start_latest.copy()
        for _ in range(K):
            g = MLP_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels_array[range_samples[m]:range_samples[m+1]], 3)
            # grads += g * n_sample_fraction[m]
            grads = [(grads[i] + g[i] * n_sample_fraction[m]) for i in range(len(x_start_latest))]
            # x -= stepsize * g
            x = [(x[i] - stepsize * g[i]) for i in range(len(x_start_latest))]
    for m in Kcache:
        x = x_start_delayed.copy()
        for _ in range(K):
            g = MLP_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels_array[range_samples[m]:range_samples[m+1]], 3)
            # grads += g * n_sample_fraction[m]
            grads = [(grads[i] + g[i] * n_sample_fraction[m]) for i in range(len(x_start_delayed))]
            # x -= stepsize * g
            x = [(x[i] - stepsize * g[i]) for i in range(len(x_start_delayed))]
#    grads
    return grads

def one_inner_outer_iteration3(x_start, M, K, stepsize):
    grads = [np.zeros_like(x_start[i]) for i in range(len(x_start))]
    for m in range(M):
        x = x_start.copy()
        for _ in range(K):
            g = fedprox_MLP_stochastic_gradient(x, features[range_samples[m]:range_samples[m+1],:], labels_array[range_samples[m]:range_samples[m+1]], 3, x_start)
            # grads += g * n_sample_fraction[m]
            grads = [(grads[i] + g[i] * n_sample_fraction[m]) for i in range(len(x_start))]
            # x -= stepsize * g
            x = [(x[i] - stepsize * g[i]) for i in range(len(x_start))]
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
    iterates = [create_weight_matrices()]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        direction = one_inner_outer_iteration(iterates[-1], Kset, K, inner_stepsize)
        #iterates.append(iterates[-1] - outer_stepsize * direction)
        current_iterate = iterates[-1]
        iterates.append([(current_iterate[i] - outer_stepsize * direction[i]) for i in range(len(current_iterate))])
        if (r+1) % loss_freq == 0:
            loss, acc = evaluate(iterates[-1], features, labels_array)
            losses.append(loss)
            accuracies.append(acc)
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r + 1, R, losses[-1]), end='')
    print('')
    return [losses, accuracies]

def inner_outer_sgd2(x0_len, Kcache, Kserver, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
    losses = []
    accuracies = []
    # iterates = [np.zeros((x0_len, C))]
    iterates = [create_weight_matrices()]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        if r>1:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-2], Kcache, Kserver, K, inner_stepsize)
        else:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-1], Kcache, Kserver, K, inner_stepsize)
        current_iterate = iterates[-1]
        iterates.append([(current_iterate[i] - outer_stepsize * direction[i]) for i in range(len(current_iterate))])
        if (r+1) % loss_freq == 0:
            loss, acc = evaluate(iterates[-1], features, labels_array)
            losses.append(loss)
            accuracies.append(acc)
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r + 1, R, losses[-1]), end='')
    print('')
    return [losses, accuracies]



def inner_outer_sgd3(x0_len, M, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
    losses = []
    accuracies = []
    # iterates = [np.zeros((x0_len, C))]
    iterates = [create_weight_matrices()]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        direction = one_inner_outer_iteration3(iterates[-1], M, K, inner_stepsize)
        current_iterate = iterates[-1]
        iterates.append([(current_iterate[i] - outer_stepsize * direction[i]) for i in range(len(current_iterate))])
        if (r+1) % loss_freq == 0:
            loss, acc = evaluate(iterates[-1], features, labels_array)
            losses.append(loss)
            accuracies.append(acc)
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r + 1, R, losses[-1]), end='')
    print('')
    return [losses, accuracies]

def inner_outer_sgd4(x0_len, M, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
    losses = []
    accuracies = []
    # iterates = [np.zeros((x0_len, C))]
    iterates = [create_weight_matrices()]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        if r>1:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-2], Kcache_list[r//10], Kserver_list[r//10], K, inner_stepsize)
        else:
        	direction = one_inner_outer_iteration2(iterates[-1], iterates[-1], Kcache_list[r//10], Kserver_list[r//10], K, inner_stepsize)
        current_iterate = iterates[-1]
        iterates.append([(current_iterate[i] - outer_stepsize * direction[i]) for i in range(len(current_iterate))])
        if (r+1) % loss_freq == 0:
            loss, acc = evaluate(iterates[-1], features, labels_array)
            losses.append(loss)
            accuracies.append(acc)
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r + 1, R, losses[-1]), end='')
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
    n_reps = 1

    if alg_sel == 'delayed':
        print('Doing Partial Delayed Local SGD...')
        delay_results = np.zeros((R // loss_freq, len(tt_stepsizes)))
        delay_results_acc = np.zeros((R // loss_freq, len(tt_stepsizes)))
        for i, stepsize in enumerate(tt_stepsizes):
            print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(tt_stepsizes)))
            for rep in range(n_reps):
                [losses, accuracies] = local_sgd_delayed(x0_len, Kcache, Kserver, K, R, stepsize, loss_freq)
                delay_results[:, i] += (losses - fstar) / n_reps
                delay_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps
        delay_l = np.min(delay_results, axis=1)
        delay_a = np.max(delay_results_acc, axis=1)
        np.savez(savearray, delay_l=delay_l, delay_a=delay_a)
    elif alg_sel == 'fedprox':
        print('Doing FedProx...')
        prox_results = np.zeros((R // loss_freq, len(lc_stepsizes)))
        prox_results_acc = np.zeros((R // loss_freq, len(lc_stepsizes)))
        for i, stepsize in enumerate(lc_stepsizes):
            print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(lc_stepsizes)))
            for rep in range(n_reps):
                [losses, accuracies] = fedprox(x0_len, M, K, R, stepsize, loss_freq)
                prox_results[:, i] += (losses - fstar) / n_reps
                prox_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps
        prox_l = np.min(prox_results, axis=1)
        prox_a = np.max(prox_results_acc, axis=1)
        np.savez(savearray, prox_l=prox_l, prox_a=prox_a)
    elif alg_sel == 'random':
        print('Doing Random Delayed Local SGD...')
        random_results = np.zeros((R // loss_freq, len(tt_stepsizes)))
        random_results_acc = np.zeros((R // loss_freq, len(tt_stepsizes)))
        for i, stepsize in enumerate(tt_stepsizes):
            print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(tt_stepsizes)))
            for rep in range(n_reps):
                [losses, accuracies] = local_sgd_delayed_random(x0_len, M, K, R, stepsize, loss_freq)
                random_results[:, i] += (losses - fstar) / n_reps
                random_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps
        random_l = np.min(random_results, axis=1)
        random_a = np.max(random_results_acc, axis=1)
        np.savez(savearray, random_l=random_l, random_a=random_a)
    elif alg_sel == 'straggler':
        print('Doing Local SGD with less workers...')
        less_results = np.zeros((R//loss_freq, len(lg_stepsizes)))
        less_results_acc = np.zeros((R // loss_freq, len(lg_stepsizes)))
        for i,stepsize in enumerate(lg_stepsizes):
            print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i+1, len(lg_stepsizes)))
            for rep in range(n_reps):
                [losses, accuracies] = local_sgd(x0_len, Kserver, K, R, stepsize, loss_freq)
                less_results[:, i] += (losses - fstar) / n_reps
                less_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps
        less_l = np.min(less_results, axis=1)
        less_a = np.max(less_results_acc, axis=1)
        np.savez(savearray, less_l=less_l, less_a=less_a)
    elif alg_sel == 'fedavg':
        print('Doing Local SGD...')
        local_results = np.zeros((R // loss_freq, len(lc_stepsizes)))
        local_results_acc = np.zeros((R // loss_freq, len(lc_stepsizes)))
        for i, stepsize in enumerate(lc_stepsizes):
            print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(lc_stepsizes)))
            for rep in range(n_reps):
                [losses, accuracies] = local_sgd(x0_len, range(M), K, R, stepsize, loss_freq)
                local_results[:, i] += (losses - fstar) / n_reps
                local_results_acc[:, i] += (accuracies - np.float64(0)) / n_reps
        local_l = np.min(local_results, axis=1)
        local_a = np.max(local_results_acc, axis=1)
        np.savez(savearray, local_l=local_l, local_a = local_a)

##################################################################################################################

alg_sel = sys.argv[1] if len(sys.argv) > 1 else 'delayed'
learning_rate = sys.argv[2] if len(sys.argv) > 2 else 'grid_search'
distribution = sys.argv[3] if len(sys.argv) > 3 else 'het'
dataset = sys.argv[4] if len(sys.argv) > 4 else 'real'
plot_prefix = sys.argv[5] if len(sys.argv) > 5 else 'plots2'
dataset_name = sys.argv[6] if len(sys.argv) > 6 else 'fmnist'
print([alg_sel, learning_rate, distribution, dataset, plot_prefix, dataset_name])


#generate data
C = 10 # number of classes
# number of samples on each device follows a power law
M = 200
# n_sample = 100*np.random.zipf(2, M)
# np.savez('n_sample_more', n_sample = n_sample)
# npzfile = np.load('n_sample_balance_less.npz')
npzfile = np.load('n_sample_large.npz')
# if dataset_name == 'mnist.scale':
#     # npzfile = np.load('n_sample_mnist.npz')
#     npzfile = np.load('n_sample_balance_less.npz')
# elif dataset_name == 'cifar10':
#     npzfile = np.load('n_sample_mnist.npz')
n_sample = npzfile['n_sample']
n_sample = [n_sample[i]//5 for i in range(M)]
range_samples = [sum(n_sample[0:i]) for i in range(len(n_sample)+1)]
n_sample_fraction = n_sample/sum(n_sample)

# npzfile = np.load('random_delay_balance_less.npz')
npzfile = np.load('random_delay_large.npz')
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
# sample_min = [x for x in range(M) if n_sample[x] == 5]
# Kcache = sample_min[0:5]
# Kserver = [x for x in range(M) if x not in Kcache]


# training data
N = 60000
if dataset_name == 'mnist.scale':
    features = extract_data('train-images-idx3-ubyte-mnist.gz', N, 28)
    labels_array = extract_labels('train-labels-idx1-ubyte-mnist.gz', N).reshape(N, 1)
else:
    features = extract_data('train-images-idx3-ubyte.gz', N, 28)
    labels_array = extract_labels('train-labels-idx1-ubyte.gz', N).reshape(N, 1)
features = features/255
labels_array = np.squeeze(labels_array)
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
features_t, labels_t = features, labels_array
print(labels_array)
# generate stepsizes
if learning_rate == 'grid_search':
    n_stepsizes = 2
    tt_stepsizes = [np.exp(exponent) for exponent in np.linspace(-4, -2.5, n_stepsizes)]
    lg_stepsizes = [np.exp(exponent) for exponent in np.linspace(-4, -2.5, n_stepsizes)]
    lc_stepsizes = [np.exp(exponent) for exponent in np.linspace(-4, -2.5, n_stepsizes)]
else:
    tt_stepsizes = [float(learning_rate)]
    lg_stepsizes = [float(learning_rate)]
    lc_stepsizes = [float(learning_rate)]


x0_len = features.shape[1]
#fstar = gradient_descent(x0_len)
fstar = np.float64(0)
print('Fstar = {:.5f}'.format(fstar))

#experiment(M=10,K=5,R=200,plotfile= plot_prefix + '/smallerM-smallK-' + distribution + '.png',savearray=plot_prefix + '/smallerM-smallK-' + distribution)
experiment(M,K=5,R=200,plotfile= plot_prefix + '/smallM-smallK-' + distribution + dataset_name + alg_sel + str(learning_rate) + 'large-new-CCN.png',savearray=plot_prefix + '/smallM-smallK-' + distribution + dataset_name + alg_sel + str(learning_rate) + 'large-new-CCN')
#experiment(M=500,K=5,R=200,plotfile= plot_prefix + '/bigM-smallK-' + distribution + '.png',savearray=plot_prefix + '/bigM-smallK-' + distribution)
#experiment(M=50,K=40,R=200,plotfile= plot_prefix + '/smallM-bigK-' + distribution + '.png',savearray=plot_prefix + '/smallM-bigK-' + distribution)
#experiment(M=500,K=40,R=100,plotfile='plots/bigM-bigK-het.png',savearray='plots/bigM-bigK-het')
#experiment(M=50,K=200,R=100,plotfile='plots/smallM-biggerK-het.png',savearray='plots/smallM-biggerK-het')
