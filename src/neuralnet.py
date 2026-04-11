import numpy as np
import matplotlib as mpt

#sigmoid transformation for R-->[0,1]    
def sigmoid(out):
    return 1.0/(1.0+np.exp(-out))