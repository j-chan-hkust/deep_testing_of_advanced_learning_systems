import argparse
import sys
import numpy as np
import itertools

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#these functions made to solve memory issues
#especially important, when consiering the fact that other models will have larger inputs.
def memEfficientDot(x,y):
    if(not len(x) == len(y)):
        print("Error in memEfficientDot: arrays being multiplied are not the same length!")
        return

    index = 0
    prevIndex = 0
    sum = 0
    end = False
    while(not end):
        index += 1000

        if(index >= len(x)): #if we exceed the vector, then we shorten
            index = len(x)-1
            end = True

        sum += np.dot(x[prevIndex:index], y[prevIndex:index]) #haha this is true lazy
        prevIndex = index
    return sum

#the input of this matrix must be in stacked arrays of the variables
#i.e. [[x1,x2,x3,....xn]
#       [y1,y2,y3,...yn]
#       [z1,z2,z3,...zn]]

def memEfficientCov(input_matrix, mean):
    #create kxk array, where k is num of variables. this is the cov. array
    covArray = np.zeros(shape = (len(input_matrix),len(input_matrix)))
    for i in range(len(input_matrix)):
        for j in range(len(input_matrix)):
            if i==j:
                covArray[i][j] = np.var(input_matrix[i]) #haha im cheating again
            else:
                covArray[i][j] = memEfficientDot(input_matrix[i],input_matrix[j])/len(input_matrix[i])-mean[i]*mean[j]
                #print(covArray[i][j])
            # elif j>i: #we are in the upperright triangle
            #     covArray[i][j] = memEfficientDot(input_matrix[i],input_matrix[j])/len(input_matrix[i])-mean[i]*mean[j]
            #     print(covArray[i][j])
            # elif j<i:
            #     covArray[j][i] = covArray[i][j]
    return covArray



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Collector for neuron outputs')
    parser.add_argument('neuron_output_data_filepath', type=str, help='where is the neuron output data?')
    parser.add_argument('k', type=int, help='groups of size k will be analyzed')
    args = parser.parse_args()

    k = args.k
    #handle the edge case
    if(k<0):
        sys.exit()

    neuronOutput = np.load(args.neuron_output_data_filepath) # load then numpy array
    neuronOutput = np.transpose(neuronOutput) # the neuron output needs to be transposed for covariance calculation

    if(k==1): #this means we just want the normal stat dist

        sns.distplot(neuronOutput[:,53], hist=True, kde=True,
             bins=int(40), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

        # Add labels
        plt.title('Histogram of 54th Neurons output')
        plt.xlabel('output')
        plt.ylabel('occurences')
        plt.show()

    if(k>1):
        mean = np.mean(neuronOutput, axis=1)
        cov = memEfficientCov(neuronOutput, mean)
        print(mean)
        print(len(mean))
        print(cov)
        print(cov.shape)

        np.save(args.neuron_output_data_filepath+"covarianceArray",cov)
        np.save(args.neuron_output_data_filepath+"mean",mean)
        #kgroups = itertools.combinations(range(len(neuronOutput[0])), k)
        # for group in kgroups:
        #     dataset = np.zeros(shape = (k, len(neuronOutput[:,0]))) # initialize the empty array
        #
        #     for index in group:
        #         dataset[index:] = neuronOutput[:,index] # fill in the correct index into the array
        #
        #     mean = np.mean(dataset, axis=0)
        #     cov = memEfficientCov(dataset, mean)
        #     print(group)
        #     print(mean)
        #     print(cov)
