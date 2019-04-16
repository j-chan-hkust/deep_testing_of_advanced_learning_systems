import argparse
import sys
import numpy as np
import itertools

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('classic')
import numpy as np
import pandas as pd


#run as python3 VGG16_stat_visualization.py VGG16_-6th_layer_data.npy 96 494

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Collector for neuron outputs')
    parser.add_argument('neuron_output_data_filepath', type=str, help='where is the neuron output data?')
    parser.add_argument('neuron1', type=int, help='1st neuron to analyze')
    parser.add_argument('neuron2', type=int, help='2nd neuron to analyze')
    args = parser.parse_args()

    neuronOutput = np.load(args.neuron_output_data_filepath)
    neuronOutput = np.transpose(neuronOutput) # the neuron output needs to be transposed for covariance calculation


    #dual variable distribution
    data = np.column_stack((neuronOutput[args.neuron1],neuronOutput[args.neuron2]))
    data = pd.DataFrame(data, columns=['neuron ' + str(args.neuron1) + ' stat distribution' , 'neuron ' + str(args.neuron2) + ' stat distribution'])

    with sns.axes_style('white'):
        sns.jointplot('neuron ' + str(args.neuron1) + ' stat distribution' , 'neuron ' + str(args.neuron2) + ' stat distribution', data, kind='kde');

    plt.savefig(fname = 'Distribution of ' + str(args.neuron1) + ' and ' + str(args.neuron2) + ' Neurons output')
    plt.clf()

    #single variable distribution
    sns.distplot(neuronOutput[args.neuron1], hist=True, kde=True,
         bins=int(40), color = 'darkblue',
         hist_kws={'edgecolor':'black'},
         kde_kws={'linewidth': 4})

    plt.title('Histogram of ' + str(args.neuron1) + ' Neurons output')
    plt.xlabel('output')
    plt.ylabel('occurences')
    plt.savefig('Histogram of ' + str(args.neuron1) + ' Neurons output')
    plt.clf()

    sns.distplot(neuronOutput[args.neuron2], hist=True, kde=True,
         bins=int(40), color = 'darkblue',
         hist_kws={'edgecolor':'black'},
         kde_kws={'linewidth': 4})

    plt.title('Histogram of ' + str(args.neuron2) + ' Neurons output')
    plt.xlabel('output')
    plt.ylabel('occurences')
    plt.savefig('Histogram of ' + str(args.neuron2) + ' Neurons output')
