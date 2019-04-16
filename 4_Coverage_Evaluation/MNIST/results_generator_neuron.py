from __future__ import print_function

import argparse
import keras
import os
import sys

from keras import models
from keras.models import load_model, Model
from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.style.use('classic')

from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#calculates the orthant coverage of a certain dataset
def calculate_nth_layer_orthant_coverage(model, test_corpus, model_layer_dict, layer, mean_vector, covariance_matrix, group_size, threshold):
    shortened_model = create_shortened_model(model, layer)

    for input_path in test_corpus:
        #load image
        input = preprocess_image(input_path)
        #calculate the covereage by updating a layer_output
        update_orthant_coverage(input, shortened_model, model_layer_dict,
            mean_vector, covariance_matrix, group_size, threshold)

    return get_orthant_coverage(model_layer_dict)

def calculate_neuron_coverge(model, test_corpus, model_layer_dict, threshold):
    for input_path in test_corpus:
        #load image
        input = preprocess_image(input_path)

        update_neuron_coverage(input, model, model_layer_dict, threshold)

    return get_neuron_coverage(model_layer_dict)

if __name__ == "__main__":

    model_path = "cifar10vgg.h5"
    try:#load mean vector and cov array
        mean_vector = np.load(mean_vector_path)
        covariance_matrix = np.load(covariance_matrix_path)
    except:
        print("FileLoad Error: cannot load mean vector or covariance matrix array")
        sys.exit()
    inputs_path = "inputs"
    threshold = 0.7
    group_size = 1
    model_name = "leNet5"
    attack_name = "FGSM"

    model = keras.models.load_model(model_path)
    corpus_dir = inputs_path + '/base_set'
    corpus_paths = [os.path.join(corpus_dir,dir) for dir in os.listdir(corpus_dir)]
    corpus_len = len(corpus_paths)

    base_model_layer_dict = init_neuron_coverage_table(model)
    #this vector will be used to plot a graph later
    initial_coverage_vector = [calculate_neuron_coverge(model, corpus_paths, base_model_layer_dict, threshold)[2]]
    model_layer_dict = deepcopy(base_model_layer_dict) #make a deepcopy
    coverage_vector = deepcopy(initial_coverage_vector)
    print("initial coverage is: " + str(coverage_vector))
    print(initial_coverage_vector)

    corpus_dir = inputs_path+'/'+model_name+'/'+attack_name
    corpus_paths = [os.path.join(corpus_dir,dir) for dir in os.listdir(corpus_dir)]
    corpus_len = len(corpus_paths)

    coverage_data = pd.DataFrame({"coverage":[]}) #empty dataframe
    for i in range(5):
        #randomize the corpus paths
        random.seed(i)
        corpus_paths = random.sample(corpus_paths, len(corpus_paths))
        print(corpus_paths)

        #gradually update the vector (which we will plot)
        for corpus_path in corpus_paths:
            input = preprocess_image(corpus_path)
            update_neuron_coverage(input, model, model_layer_dict, threshold)
            coverage_vector.append(get_neuron_coverage(model_layer_dict)[2])
        coverage_data = coverage_data.append(pd.DataFrame({'adversarial images added':range(len(coverage_vector)),"coverage":coverage_vector}))
        coverage_vector = deepcopy(initial_coverage_vector)

    np.save(model_name+"_"+attack_name+"_neuron"+"_threshold_"+str(threshold).replace('.',',')+"_group_size_"+str(group_size),np.array(coverage_vector))
    sns.lineplot(x="adversarial images added",y="coverage",data=coverage_data.reset_index())
    plt.savefig("graph of "+model_name+"_"+attack_name+"_neuron"+"_threshold_"+str(threshold).replace('.',',')+"_group_size_"+str(group_size))
    plt.clf()





    attack_name = "bim"
    corpus_dir = inputs_path+'/'+model_name+'/'+attack_name
    corpus_paths = [os.path.join(corpus_dir,dir) for dir in os.listdir(corpus_dir)]
    corpus_len = len(corpus_paths)

    model_layer_dict = deepcopy(base_model_layer_dict)
    coverage_data = pd.DataFrame({"coverage":[]}) #empty dataframe
    for i in range(5):
        #randomize the corpus paths
        random.seed(i)
        corpus_paths = random.sample(corpus_paths, len(corpus_paths))
        print(corpus_paths)

        #gradually update the vector (which we will plot)
        for corpus_path in corpus_paths:
            input = preprocess_image(corpus_path)
            update_neuron_coverage(input, model, model_layer_dict, threshold)
            coverage_vector.append(get_neuron_coverage(model_layer_dict)[2])
        coverage_data = coverage_data.append(pd.DataFrame({'adversarial images added':range(len(coverage_vector)),"coverage":coverage_vector}))
        coverage_vector = deepcopy(initial_coverage_vector)

    np.save(model_name+"_"+attack_name+"_neuron"+"_threshold_"+str(threshold).replace('.',',')+"_group_size_"+str(group_size),np.array(coverage_vector))
    sns.lineplot(x="adversarial images added",y="coverage",data=coverage_data.reset_index())
    plt.savefig("graph of "+model_name+"_"+attack_name+"_neuron"+"_threshold_"+str(threshold).replace('.',',')+"_group_size_"+str(group_size))
    plt.clf()

    attack_name = "cwl2"
    corpus_dir = inputs_path+'/'+model_name+'/'+attack_name
    corpus_paths = [os.path.join(corpus_dir,dir) for dir in os.listdir(corpus_dir)]
    corpus_len = len(corpus_paths)

    model_layer_dict = deepcopy(base_model_layer_dict)
    coverage_data = pd.DataFrame({"coverage":[]}) #empty dataframe
    for i in range(5):
        #randomize the corpus paths
        random.seed(i)
        corpus_paths = random.sample(corpus_paths, len(corpus_paths))
        print(corpus_paths)

        #gradually update the vector (which we will plot)
        for corpus_path in corpus_paths:
            input = preprocess_image(corpus_path)
            update_neuron_coverage(input, model, model_layer_dict, threshold)
            coverage_vector.append(get_neuron_coverage(model_layer_dict)[2])
        coverage_data = coverage_data.append(pd.DataFrame({'adversarial images added':range(len(coverage_vector)),"coverage":coverage_vector}))
        coverage_vector = deepcopy(initial_coverage_vector)

    np.save(model_name+"_"+attack_name+"_neuron"+"_threshold_"+str(threshold).replace('.',',')+"_group_size_"+str(group_size),np.array(coverage_vector))
    sns.lineplot(x="adversarial images added",y="coverage",data=coverage_data.reset_index())
    plt.savefig("graph of "+model_name+"_"+attack_name+"_neuron"+"_threshold_"+str(threshold).replace('.',',')+"_group_size_"+str(group_size))
    plt.clf()

    attack_name = "smm"
    corpus_dir = inputs_path+'/'+model_name+'/'+attack_name
    corpus_paths = [os.path.join(corpus_dir,dir) for dir in os.listdir(corpus_dir)]
    corpus_len = len(corpus_paths)

    model_layer_dict = deepcopy(base_model_layer_dict)
    coverage_data = pd.DataFrame({"coverage":[]}) #empty dataframe
    for i in range(5):
        #randomize the corpus paths
        random.seed(i)
        corpus_paths = random.sample(corpus_paths, len(corpus_paths))
        print(corpus_paths)

        #gradually update the vector (which we will plot)
        for corpus_path in corpus_paths:
            input = preprocess_image(corpus_path)
            update_neuron_coverage(input, model, model_layer_dict, threshold)
            coverage_vector.append(get_neuron_coverage(model_layer_dict)[2])
        coverage_data = coverage_data.append(pd.DataFrame({'adversarial images added':range(len(coverage_vector)),"coverage":coverage_vector}))
        coverage_vector = deepcopy(initial_coverage_vector)

    np.save(model_name+"_"+attack_name+"_neuron"+"_threshold_"+str(threshold).replace('.',',')+"_group_size_"+str(group_size),np.array(coverage_vector))
    sns.lineplot(x="adversarial images added",y="coverage",data=coverage_data.reset_index())
    plt.savefig("graph of "+model_name+"_"+attack_name+"_neuron"+"_threshold_"+str(threshold).replace('.',',')+"_group_size_"+str(group_size))
    plt.clf()

    # model_layer_dict = deepcopy(base_model_layer_dict)
    # coverage_vector = deepcopy(initial_coverage_vector)
    # corpus_dir = inputs_path+'/leNet5/bim'
    # corpus_paths = [os.path.join(corpus_dir,dir) for dir in os.listdir(corpus_dir)]
    # corpus_len = len(corpus_paths)
    # #randomize the corpus paths
    # random.seed(1234)
    # corpus_paths = random.sample(corpus_paths, len(corpus_paths))
    #
    # #gradually update the vector (which we will plot)
    # for corpus_path in corpus_paths:
    #     input = preprocess_image(corpus_path)
    #     update_orthant_coverage(input, shortened_model, model_layer_dict, mean_vector, covariance_matrix, group_size, threshold)
    #     coverage_vector.append(get_orthant_coverage(model_layer_dict)[2])
    #
    # np.save("leNet5_bim_orthant_layer_"+str(layer_index)+"_threshold_"+str(threshold)+"_group_size_"+str(group_size),np.array(coverage_vector))
    # plt.plot(coverage_vector)
    # plt.xlabel('adversarial images added')
    # plt.ylabel('neuron coverage')
    # plt.savefig("graph of leNet5_bim_orthant_layer_"+str(layer_index)+"_threshold_"+str(threshold)+"_group_size_"+str(group_size))
    # plt.clf()
    #
    # model_layer_dict = deepcopy(base_model_layer_dict)
    # coverage_vector = deepcopy(initial_coverage_vector)
    # corpus_dir = inputs_path+'/leNet5/smm'
    # corpus_paths = [os.path.join(corpus_dir,dir) for dir in os.listdir(corpus_dir)]
    # corpus_len = len(corpus_paths)
    # #randomize the corpus paths
    # random.seed(1234)
    # corpus_paths = random.sample(corpus_paths, len(corpus_paths))
    #
    # #gradually update the vector (which we will plot)
    # for corpus_path in corpus_paths:
    #     input = preprocess_image(corpus_path)
    #     update_orthant_coverage(input, shortened_model, model_layer_dict, mean_vector, covariance_matrix, group_size, threshold)
    #     coverage_vector.append(get_orthant_coverage(model_layer_dict)[2])
    #
    # np.save("leNet5_smm_orthant_layer_"+str(layer_index)+"_threshold_"+str(threshold)+"_group_size_"+str(group_size),np.array(coverage_vector))
    # plt.plot(coverage_vector)
    # plt.xlabel('adversarial images added')
    # plt.ylabel('neuron coverage')
    # plt.savefig("graph of leNet5_smm_orthant_layer_"+str(layer_index)+"_threshold_"+str(threshold)+"_group_size_"+str(group_size))
    # plt.clf()
    #
    # model_layer_dict = deepcopy(base_model_layer_dict)
    # coverage_vector = deepcopy(initial_coverage_vector)
    # corpus_dir = inputs_path+'/leNet5/cwl2'
    # corpus_paths = [os.path.join(corpus_dir,dir) for dir in os.listdir(corpus_dir)]
    # corpus_len = len(corpus_paths)
    # #randomize the corpus paths
    # random.seed(1234)
    # corpus_paths = random.sample(corpus_paths, len(corpus_paths))
    #
    # #gradually update the vector (which we will plot)
    # for corpus_path in corpus_paths:
    #     input = preprocess_image(corpus_path)
    #     update_orthant_coverage(input, shortened_model, model_layer_dict, mean_vector, covariance_matrix, group_size, threshold)
    #     coverage_vector.append(get_orthant_coverage(model_layer_dict)[2])
    #
    # np.save("leNet5_cwl2_orthant_layer_"+str(layer_index)+"_threshold_"+str(threshold)+"_group_size_"+str(group_size),np.array(coverage_vector))
    # plt.plot(coverage_vector)
    # plt.xlabel('adversarial images added')
    # plt.ylabel('neuron coverage')
    # plt.savefig("graph of leNet5_cwl2_orthant_layer_"+str(layer_index)+"_threshold_"+str(threshold)+"_group_size_"+str(group_size))
