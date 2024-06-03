import numpy as np
import os

from src.config import config
 
class preprocess_data:

    def fit(self,x,y=None):

        self.num_rows = x.shape[0]
        self.num_input_features = x.shape[1]

        if len(y.shape) == 1:
            self.target_feature_dim = 1
        else:
            self.target_feature_dim = y.shape[1]

    def transform(self,x=None,y=None):

        self.x = np.array(x)
        self.y = np.array(y).reshpe(self.num_rows,)

        return self.x ,self.y