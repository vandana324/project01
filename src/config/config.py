import pathlib
import os # operation related file system done by this os library
import src

NUM_INPUTS = 2
NUM_LAYERS = 3
P = [NUM_INPUTS , 2, 1]

#config always store constant that we will use in our project

f = [None,"Linear","sigmoid"]
LOSS_FUNCTION = "Mean Squared Error"
MINI_BATCH_SIZE = 1

PACKAGE_ROOT = pathlib.path(src.__file__).resolve().parent # here we are coverting src file with root
 

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")  # this is used to set location of dataset basically link path to config 
#"/src/datasets"
SAVED_MODEL_PATH =  os.path.join(PACKAGE_ROOT,"trained_models")
"""theta0 = [None]
theta = [None]

z = [None]*NUM_LAYERS
h = [None]*NUM_LAYERS


del_fl_by_del_z = [None]*NUM_LAYERS
del_hl_by_del_theta0 = [None]*NUM_LAYERS
del_hl_by_del_theta = [None]*NUM_LAYERS
del_L_by_del_h = [None]*NUM_LAYERS
del_L_by_del_theta0 = [None]*NUM_LAYERS
del_L_by_del_theta = [None]*NUM_LAYERS"""