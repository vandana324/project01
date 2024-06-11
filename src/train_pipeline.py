import pandas as pd
import numpy as np

from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_datsets , save_model, load_model

import pipeline as pl



theta0 = [None]
theta = [None]

z = [None]*config.NUM_LAYERS
h = [None]*config.NUM_LAYERS


del_fl_by_del_z = [None]*config.NUM_LAYERS
del_hl_by_del_theta0 = [None]*config.NUM_LAYERS
del_hl_by_del_theta = [None]*config.NUM_LAYERS
del_L_by_del_h = [None]*config.NUM_LAYERS
del_L_by_del_theta0 = [None]*config.NUM_LAYERS
del_L_by_del_theta = [None]*config.NUM_LAYERS

def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):

    return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs,current_layer_neurons_weights)


def initialize_layer_weights(num_units_l_1, num_units_l):

  return np.random.uniform(low=-1,high=1,size=(num_units_l_1,num_units_l))


def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):

  if current_layer_neurons_activation_function == "linear":
    return current_layer_neurons_weighted_sums

  elif current_layer_neurons_activation_function == "sigmoid":
    return 1/(1 + np.exp(-current_layer_neurons_weighted_sums))

  elif current_layer_neurons_activation_function == "tanh":
    return (np.exp(current_layer_neurons_weighted_sums) - np.exp(-current_layer_neurons_weighted_sums))/ \
            (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))

  elif current_layer_neurons_activation_function == "relu":
    return current_layer_neurons_weighted_sums * (current_layer_neurons_weighted_sums > 0)
  

def del_layer_neurons_outputs_wrt_weighted_sums(current_layer_neurons_activation_function, current_layer_neurons_weighted_sums):

  if current_layer_neurons_activation_function == "linear":
    return np.ones_like(current_layer_neurons_weighted_sums)

  elif current_layer_neurons_activation_function == "sigmoid":
    current_layer_neurons_outputs = 1/(1 + np.exp(-current_layer_neurons_weighted_sums))
    return current_layer_neurons_outputs * (1 - current_layer_neurons_outputs)

  elif current_layer_neurons_activation_function == "tanh":
    return (2/(np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums)))**2

  elif current_layer_neurons_activation_function == "relu":
    return (current_layer_neurons_weighted_sums > 0)
  
def del_layer_neurons_outputs_wrt_biases(current_layer_neurons_outputs_dels):

  return current_layer_neurons_outputs_dels


def del_layer_neurons_outputs_wrt_weights(previous_layer_neurons_outputs,current_layer_neurons_outputs_dels):

  return np.matmul(previous_layer_neurons_outputs.T,current_layer_neurons_outputs_dels)


def run_training(tol,epsilon):


  epoch_counter = 0
  mse = 1
  loss_per_epoch = list()

  training_data = load_datsets("train.csv")

  obj = pp.preprocess_data()
  obj.fit(training_data.iloc[:,0:2],training_data.iloc[:,2])
  x_train,y_train = obj.transform(training_data.iloc[:,0:2],training_data.iloc[:,2])

  pl.initialize_parameters()

  while mse > tol:
     mse = 0 
    
     for i in range(x_train.shape[0]):
       
         h[0] = x_train[i].reshape(1,x_train.shape[1])

         for l in range(1,config.NUM_LAYERS):
           
              z[1] = layer_neurons_weighted_sum(h[l-1],pl.theta0[1],pl.theta[1])

              h[1] = layer_neurons_output(z[1],config.f[1])

              del_fl_by_del_z[l] = del_layer_neurons_outputs_wrt_weighted_sums(f[l],z[l])
               #print("del_fl_by_del_z[{}].shape = {}".format(l,del_fl_by_del_z[l].shape))
              del_hl_by_del_theta0[l] = del_layer_neurons_outputs_wrt_biases(del_fl_by_del_z[l])
               #print("del_hl_by_del_theta0[{}].shape = {}".format(l,del_hl_by_del_theta0[l].shape))
              del_hl_by_del_theta[l] = del_layer_neurons_outputs_wrt_weights(h[l-1],del_fl_by_del_z[l])
      #     print("del_hl_by_del_theta[{}].shape = {}".format(l,del_hl_by_del_theta[l].shape))

       #print("\n")

         y_train[i] = y[i].reshape(y_train[i].shape[0],1)
              #print("y[{}].shape = {}".format(i,y[i].shape))
         L = (1/2)*(y[i][0] - h[config.NUM_LAYERS-1][0,0])**2
    #The above expression is the expression of Loss Function in our case which is Squared Error.

         mse = mse + L

         del_L_by_del_h[config.NUM_LAYERS-1] = (h[config.NUM_LAYERS-1] - y_train[0])
    #The above expression is the expression of derivative of the loss function with respect to the output of the Neural Network.
    #print("del_L_by_del_h[{}].shape = {}".format(NUM_LAYERS-1,del_L_by_del_h[NUM_LAYERS-1].shape))
         for l in range(1,config.NUM_LAYERS-2,0,-1):

             del_L_by_del_h[l] = np.matmul(del_L_by_del_h[l+1], (del_fl_by_del_z[l+1] * theta[l+1]).T)
      #print("del_L_by_del_h[{}].shape = {}".format(l,del_L_by_del_h[l].shape))
      #The expression shown above is computing the derivative of the Loss Function wrt to the output of neurons in a layer. For more information, please see
      #the first picture below.

    #print("\n\n\n")

    #Now, we will be running Gradient Descent Algorithm (Backpropagation Algorithm) by computing the overall derivative of the Loss Function wrt the biases and
    #weights of each layer (as shown in the second picture below) and updating these parameters.

     for l in range(1,config.NUM_LAYERS):

      del_L_by_del_theta0[l] = del_L_by_del_h[l] * del_hl_by_del_theta0[l]
      del_L_by_del_theta[l] = del_L_by_del_h[l] * del_hl_by_del_theta[l]

      pl.theta0[l] = theta0[l] - (epsilon * del_L_by_del_theta0[l])
      pl.theta[l] = theta[l] - (epsilon *del_L_by_del_theta[l])

     mse = mse/x_train.shape[0]
  epoch_counter = epoch_counter + 1
  loss_per_epoch.append(mse)

  print("Epoch # {}, Loss = {}".format(epoch_counter,mse))




if __name__=="__main__":
  run_training(10**(-3),10**(-7))
  save_model(pl.theta0,pl.theta)