from __future__ import print_function
import tensorflow as tf
import numpy as np
print(tf.__version__)

import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model,Sequential
from keras.datasets import cifar10
from keras.models import load_model
from keras.losses import sparse_categorical_crossentropy
import os
import scipy.sparse
import argparse
import network_structure
import defined_losses

img_rows = 32
img_cols = 32
input_channel = 3
input_shape = (img_rows, img_cols, input_channel)
EPS = 0.00001

def unbalanced_cifar10(minor_cls, minor_ratio, oversample = 20):
    cifar10 = keras.datasets.cifar10
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, input_channel)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, input_channel)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    new_xtrain = np.array([]).reshape((0,img_rows,img_cols,input_channel))
    new_ytrain = np.array([])
    fixed_label = np.array([],dtype=np.int64)
    remained_label = np.array([],dtype=np.int64)
    out_labels = np.array([])
    out_train = np.array([]).reshape((0,img_rows,img_cols,input_channel))
    for label in range(10):
        ratio = minor_ratio
        if label not in minor_cls:
            ratio = 0.9
        label_inds = np.where(y_train==label)
        sliced = int(len(label_inds[0])*ratio)        
        fixed_label = np.concatenate((fixed_label,label_inds[0][:sliced]), axis = 0)
        remained_label = np.concatenate((remained_label,label_inds[0][sliced:]), axis = 0)
        if label in minor_cls:
            for rep in range(oversample):
                out_train = np.concatenate((out_train,x_train[label_inds[0][:sliced],:]), axis = 0)
                out_labels = np.concatenate((out_labels,y_train[label_inds[0][:sliced]]), axis = 0)
            new_xtrain = np.concatenate((new_xtrain,x_train[label_inds[0][:sliced],:]), axis = 0)
            new_ytrain = np.concatenate((new_ytrain,y_train[label_inds[0][:sliced]]), axis = 0)            
        else:
            new_xtrain = np.concatenate((new_xtrain,x_train[label_inds[0][:sliced],:]), axis = 0)
            new_ytrain = np.concatenate((new_ytrain,y_train[label_inds[0][:sliced]]), axis = 0)
    print("Imbalanced training image number: " + str(len(new_xtrain))) 
    print("Unlabeled image number: " + str(len(remained_label)))
    return new_xtrain, new_ytrain, out_train, out_labels, fixed_label, remained_label, x_test, y_test, x_train, y_train

def model_initializing(model_name):
    if model_name == "simplenet":
        model = network_structure.simple_net(input_shape)
    elif model_name == "res32":
        model = network_structure.resnet_v2(input_shape, 32, num_classes=10)
    elif model_name == "res56":
        model = network_structure.resnet_v2(input_shape, 56, num_classes=10)
    return model

def validation_once(model, test_info):
    x_test, y_test = test_info
    test_predictions = model.predict(x_test)
    test_out_labels = np.argmax(test_predictions, axis=1)
    sub_accuracy = (test_out_labels == y_test)
    print("label, recall, precision, f1")
    for label in range(10):
        label_pos = np.where(y_test==label)[0]
        label_gt = float(len(label_pos))
        label_tp = float(sum(sub_accuracy[label_pos]))
        label_pred = len(np.where(test_out_labels==label)[0]) + EPS
        prec = label_tp/(label_pred + EPS)
        rec = label_tp/label_gt
        f1 = 2*prec*rec/(prec+rec+ EPS)
        print(label,label_tp/label_gt,label_tp/label_pred,f1)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Error rate:', 1.0-test_acc)
        
def select_net_train(model_name, iteration, epoch, minor_cls, data_info, test_info, single_net = False):
    new_xtrain, new_ytrain, out_train, out_labels, fixed_label, remained_label, x_train, y_train = data_info
    model1 = model_initializing(model_name)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model1.compile(optimizer=sgd, #'adam',
                  loss= 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    if not single_net:
        select_model = network_structure.selection_net(10)
        select_model.compile(optimizer='adam',
                  loss=defined_losses.multiply_loss,
                  metrics=['accuracy'])
    datagen = ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)
    EPOCH = epoch
    inc_train = out_train
    inc_labels = out_labels
    sel_epoch = 5

    for round in range(iteration):
        print("Round: "+str(round))
        model1_train = np.concatenate((new_xtrain,inc_train), axis = 0)
        model1_label = np.concatenate((new_ytrain,inc_labels), axis = 0)
        
        datagen.fit(model1_train)    
        model1.fit_generator(datagen.flow(model1_train, model1_label),
                                epochs=EPOCH,
                                validation_data=test_info,
                                workers=4)
        validation_once(model1, test_info)
        
        model1_predict = model1.predict(x_train)
        pred_y = np.argmax(model1_predict, axis=1)
        loss_y = np.zeros((len(pred_y),10))
        flat_loss_y = np.zeros((len(pred_y)))
        train_loss =  np.zeros(pred_y.shape)
        for i in range(len(pred_y)):
            flat_loss_y[i] = pred_y[i]
            loss_y[i][pred_y[i]] = 1
            if i in remained_label:
                flat_loss_y[i] = pred_y[i]
                loss_y[i][pred_y[i]] = 1
            else:
                flat_loss_y[i] = y_train[i]
                loss_y[i][y_train[i]] = 1
            
        train_loss = defined_losses.cross_entropy(model1_predict, loss_y, epsilon=1e-12)
        pred_y = np.asarray(pred_y)
        in_minor = np.in1d(pred_y, minor_cls).reshape(pred_y.shape)
        if not single_net:
            select_model.fit(model1_predict, train_loss, epochs=sel_epoch)
            selected = select_model.predict(model1_predict)
            larger = selected > 0.6
        else:
            larger = train_loss < 0.6
        new_ind = in_minor&larger.flatten()
        inc_labels = flat_loss_y[new_ind]
        inc_train = x_train[new_ind][:]
    
def oversample_train(model_name, epoch, minor_cls, data_info, test_info):
    new_xtrain, new_ytrain, out_train, out_labels, fixed_label, remained_label, x_train, y_train = data_info
    model1 = model_initializing(model_name)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model1.compile(optimizer=sgd, #'adam',
                  loss= 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    datagen = ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)
    EPOCH = epoch
    inc_train = out_train
    inc_labels = out_labels
    sel_epoch = 5

    model1_train = np.concatenate((new_xtrain,inc_train), axis = 0)
    model1_label = np.concatenate((new_ytrain,inc_labels), axis = 0)
        
    datagen.fit(model1_train)    
    model1.fit_generator(datagen.flow(model1_train, model1_label),
                            epochs=EPOCH,
                            validation_data=test_info,
                            workers=4)
    validation_once(model1, test_info)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iterative SelectNet training')
    parser.add_argument('mode', type=str, help='The training mode: oversample, SelectNet, single', choices=['oversample', 'SelectNet', 'single'])
    parser.add_argument('--minor_cls', type=int, nargs = '+', default=[0,2,6,7], help='The index of classes that are chosen as minor class')
    parser.add_argument('--minor_ratio', type=float, default = 0.01, help='The imbalanced ratio of minor classes')
    parser.add_argument('--iteration', type=int, default = 10, help='Interations of updating data')
    parser.add_argument('--epoch', type=int, default = 10, help='Training epoch for each iteration')
    parser.add_argument('--model', type=str, default = 'simplenet', choices=['simplenet', 'res32', 'res56'], help='network structure')
    args = parser.parse_args()
    new_xtrain, new_ytrain, out_train, out_labels, fixed_label, remained_label, x_test, y_test, x_train, y_train = unbalanced_cifar10(args.minor_cls, args.minor_ratio)
    if args.mode == "oversample":
        new_xtrain, new_ytrain, out_train, out_labels, fixed_label, remained_label, x_test, y_test, x_train, y_train = unbalanced_cifar10(args.minor_cls, args.minor_ratio, oversample = int(1/args.minor_ratio))
        data_info = (new_xtrain, new_ytrain, out_train, out_labels, fixed_label, remained_label, x_train, y_train)
        test_info = (x_test, y_test)
        oversample_train(args.model, args.iteration*args.epoch, args.minor_cls, data_info, test_info)
    elif args.mode == "SelectNet":
        data_info = (new_xtrain, new_ytrain, out_train, out_labels, fixed_label, remained_label, x_train, y_train)
        test_info = (x_test, y_test)
        select_net_train(args.model, args.iteration, args.epoch, args.minor_cls, data_info, test_info)
    elif args.mode == "single":
        data_info = (new_xtrain, new_ytrain, out_train, out_labels, fixed_label, remained_label, x_train, y_train)
        test_info = (x_test, y_test)
        select_net_train(args.model, args.iteration, args.epoch, args.minor_cls, data_info, test_info, single_net=True)
    
    