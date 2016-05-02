
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

import theano 
import theano.tensor as T

import os

from cnn import represent
#from rnn import trainRecNet, evaluate

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import timeit
import cPickle

import SimpleITK as sitk
import sys


from mlp import HiddenLayer, HiddenLayer2
from logistic_sgd import LogisticRegression


def generate_data(file_prefixes, p_width = 10):
    def generate_patch(img, (i, j), p_width = 10): # patch area : (10*10)
        assert p_width%2 == 0
        p_width/=2
        patch = img.crop((i-p_width, j-p_width, i+p_width, j+p_width))
        return patch

    #load images
    patch_tuple = ()
    y_list = []
    data_size = 0
    ind = 0
    print 'Data generated: 0.00 %'
    for f in file_prefixes:
        img = Image.open(open('../data/left_scaled/'+f+'.bmp'))
        label = Image.open(open('../data/labeled_scaled/'+f+'.bmp'))
        label = np.array(label, dtype='int32')
        rows, cols = img.size
        for i in range(rows):
            for j in range(cols):
                # note img(i, j) -> label(j, i)
                l = label[j][i]
                if l==0:
                    continue
                patch_10 = generate_patch(img, (i, j), p_width)
                #print patch_10
                #print np.array([patch_10], dtype='float64')
                #patch_10_ = np.array([patch_10]).astype(float64).transpose(2, 0, 1)/256
                #print np.array([np.array(patch_10, dtype='float64')])
                #print np.array([np.array(patch_10, dtype='float64'), [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]).shape
                #for p in np.array([np.array(patch_10, dtype='float64'),[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]):
                #   for row in p:
                #       print row
                #print "pwidth: " + str(p_width)
                patch_2nd_chan = np.zeros((p_width, p_width))
                patch_3rd_chan = np.zeros((p_width, p_width))
                patch_10_ = np.array([np.array(patch_10, dtype='float64'), patch_2nd_chan, patch_3rd_chan]).transpose(2, 0, 1)/256
                #for p in patch_10_:
                #   print "**************"
                #   for row in p:
                #       print row
                #sys.exit()
                patch_tuple += (patch_10_, )
                y_list.append(l-1)
                data_size += 1
        ind += 1
        print '\033[FData generated: ', ind*100./len(file_prefixes), ' %\t'

    assert len(patch_tuple) == len(y_list) and len(y_list) == data_size
    #print patch_tuple
    #print type(patch_tuple)
    #print np.concatenate(patch_tuple).shape
    #print "data_size: " + str(data_size) + ", " + "p_width: " + str(p_width)
    x = np.concatenate(patch_tuple).reshape((data_size, 3, p_width, p_width))
    y = y_list
    return x, y


def evaluate(test_x, inp_dim=90, batch_size=500, n_recurrences=4):
    #test_x, test_y = data_xy
    n_test_batches = test_x[0].get_value(borrow=True).shape[0] / batch_size

    rng = np.random.RandomState(23455)
    
    W1 = theano.shared(np.asarray(rng.uniform(low=-1., high=-1., size=(300, 300)), dtype=theano.config.floatX), borrow=True)
    U1 = theano.shared(np.asarray(rng.uniform(low=-1., high=-1., size=(90, 300)), dtype=theano.config.floatX), borrow=True)
    b1 = theano.shared(np.asarray(rng.uniform(low=-1., high=-1., size=(300,)), dtype=theano.config.floatX), borrow=True)
    W2 = theano.shared(np.asarray(rng.uniform(low=-1., high=-1., size=(300, 10)), dtype=theano.config.floatX), borrow=True)
    b2 = theano.shared(np.asarray(rng.uniform(low=-1., high=-1., size=(10, )), dtype=theano.config.floatX), borrow=True)
    

    save_file = open('rnnparams.pkl')
    W1.set_value(cPickle.load(save_file), borrow=True)
    U1.set_value(cPickle.load(save_file), borrow=True)
    b1.set_value(cPickle.load(save_file), borrow=True)
    W2.set_value(cPickle.load(save_file), borrow=True)
    b2.set_value(cPickle.load(save_file), borrow=True)
    save_file.close()

    index = T.lscalar()

    # start of theano function
    x = test_x[0][index*batch_size: (index+1)*batch_size]
    #y = test_y[index*batch_size: (index+1)*batch_size]

    layer0 = HiddenLayer(
        rng,
        input = x,
        n_in = 90,
        n_out = 300,
        W = U1,
        b = b1,
        activation = T.tanh
    )

    # TODO: should i start from 0??
    for i in range(0, n_recurrences):
        if i==0:
            inp = layer0.output
        else:
            inp = layer1.output

        layer1 = HiddenLayer2(
            rng,
            input0 = test_x[i][index*batch_size: (index+1)*batch_size],
            input_1 = inp,
            n_in0 = 90,
            n_in_1 = 300,
            n_out = 300,
            W = W1,
            U = U1,
            b = b1,
            activation = T.tanh
        )

    layer2 = LogisticRegression(input=layer1.output, n_in=300, n_out=10, W=W2, b=b2)
    
    #cost = layer2.negative_log_likelihood(y)
    
    #f = theano.function([index], cost)
    
    y_ = layer2.y_pred
    
    g = theano.function([index], y_)

    #losses = [
    #    f(ind)
    #    for ind in xrange(n_test_batches)
    #]
    #score = numpy.mean(losses)

    print "ntestbatches: " + str(n_test_batches)

    preds = [
        g(ind)
        for ind in xrange(n_test_batches)
    ]

    return preds


def shared_data_x(data_x):
    return theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)


#-------------------------
img_slice = "image_transducer0_3D_t1450441616987_layer0_s_57"

'''
p_width = 4
x, _ = generate_data([img_slice], p_width)
#test_x = theano.shared(np.asarray(x[(4*len(x)/5):len(x)], dtype=theano.config.floatX), borrow=True)
net_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
rep4 = represent(net_x[:], net_x.shape.eval()[0], p_width)
'''
p_width = 20
x, _ = generate_data([img_slice], p_width)
#x = theano.shared(np.asarray(x[(4*len(x)/5):len(x)], dtype=theano.config.floatX), borrow=True)
net_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
rep20 = represent(net_x[:], net_x.shape.eval()[0], p_width)

p_width = 24
x, _ = generate_data([img_slice], p_width)
#x = theano.shared(np.asarray(x[(4*len(x)/5):len(x)], dtype=theano.config.floatX), borrow=True)
net_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
rep24 = represent(net_x[:], net_x.shape.eval()[0], p_width)

p_width = 30
x, _ = generate_data([img_slice], p_width)
#x = theano.shared(np.asarray(x[(4*len(x)/5):len(x)], dtype=theano.config.floatX), borrow=True)
net_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
rep30 = represent(net_x[:], net_x.shape.eval()[0], p_width)

#---------------------

preds_list = []

'''
for fold in range(5):
    x4 = shared_data_x(rep4[(fold*rep4.shape[0]/5):((fold+1)*rep4.shape[0]/5)])
    x10 = shared_data_x(rep10[(fold*rep10.shape[0]/5):((fold+1)*rep10.shape[0]/5)])
    x14 = shared_data_x(rep14[(fold*rep14.shape[0]/5):((fold+1)*rep14.shape[0]/5)])
    x = (x4, x10, x14)

    with open("predictions.txt", "a") as f:
        preds = evaluate(x, n_recurrences=3)
        f.write("*"*50 + "\n\n")
        f.write("FOLD " + str(fold) + "\n\n")
        
        pred_shape = np.array(preds).shape
        for i in range(pred_shape[0]):
            preds[i] = [255 if val == 1 else 0 for val in preds[i]]
        f.write(str(preds) + "\n\n")
        preds_list += preds
'''
RANGE_CAP = 1
for fold in range(RANGE_CAP):
    x20 = shared_data_x(rep20[(fold*rep20.shape[0]/RANGE_CAP):((fold+1)*rep20.shape[0]/RANGE_CAP)])
    x24 = shared_data_x(rep24[(fold*rep24.shape[0]/RANGE_CAP):((fold+1)*rep24.shape[0]/RANGE_CAP)])
    x30 = shared_data_x(rep30[(fold*rep30.shape[0]/RANGE_CAP):((fold+1)*rep30.shape[0]/RANGE_CAP)])
    x = (x20, x24, x30)

    with open("predictions.txt", "a") as f:
        preds = evaluate(x, n_recurrences=3)
        #f.write("*"*50 + "\n\n")
        #f.write("FOLD " + str(fold) + "\n\n")
        
        pred_shape = np.array(preds).shape
        for i in range(pred_shape[0]):
            preds[i] = [255 if val == 1 else 0 for val in preds[i]]
        f.write(str(preds) + "\n\n")
        preds_list += preds

arr = np.array(preds_list)
arr.transpose()
arr.resize(288, 275)
arr = np.rot90(arr)
arr = arr[::-1]
#arr = np.append(arr[:58], np.append(np.fliplr(arr[58:115]), arr[115:], axis=0), axis=0)

#with open("output.txt", "") as f:
#    for row in arr:
#        f.write(row)
#        f.write("\n\n")
#print arr.shape
img = sitk.Cast(sitk.GetImageFromArray(arr), sitk.sitkUInt8)
sitk.WriteImage(img, "/Users/elizabethcotton/Documents/indiv/rcnn/src/images/lord_almighty3_" + img_slice + "_segd.bmp")

#---------------------


