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
    max = (0,1)
    min = (0,1)
    for i in range(len(input_matrix)):
        for j in range(len(input_matrix)):
            if i==j:
                covArray[i][j] = np.var(input_matrix[i]) #haha im cheating again
            else:
                covArray[i][j] = memEfficientDot(input_matrix[i],input_matrix[j])/len(input_matrix[i])-mean[i]*mean[j]
                if covArray.item(max) < covArray.item((i,j)):
                    max = (i,j)
                if covArray.item(min) > covArray.item((i,j)):
                    min = (i,j)
                #print(covArray[i][j])
            # elif j>i: #we are in the upperright triangle
            #     covArray[i][j] = memEfficientDot(input_matrix[i],input_matrix[j])/len(input_matrix[i])-mean[i]*mean[j]
            #     print(covArray[i][j])
            # elif j<i:
            #     covArray[j][i] = covArray[i][j]
    return covArray, max, min



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Collector for neuron outputs')
    #parser.add_argument('neuron_output_data_filepath', type=str, help='where is the neuron output data?')
    args = parser.parse_args()

    neuronOutput = np.load("VGG16_-6th_layer_data.npy") # load then numpy array
    neuronOutput = np.transpose(neuronOutput) # the neuron output needs to be transposed for covariance calculation

    mean = np.mean(neuronOutput, axis=1)
    cov, maxcov, mincov = memEfficientCov(neuronOutput, mean)
    print("mean of neuron output, mean array datashape")
    print(mean)
    print(len(mean))

    print("covariance of neuron output, covariance array datashape")
    print(cov)
    print(cov.shape)

    print("neurons with maximum covariance, covariance of these neurons")
    print(maxcov)
    print(cov[maxcov[0]][maxcov[1]])

    print("neurons with minimum covariance, covariance of these neurons")
    print(mincov)
    print(cov[mincov[0]][mincov[1]])

    np.save("VGG16_-6th_layer_data.npy"+"covarianceArray",cov)
    np.save("VGG16_-6th_layer_data.npy"+"mean",mean)
