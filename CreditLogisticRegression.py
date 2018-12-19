import numpy as np
import pandas as pd
import StatFunction as sf
import math
import scipy.stats
import Projection

def custom(predicts = features = []):
	data = pd.read_csv('Credit.csv')
	data['free'] = pd.Series(np.ones(data.shape[0]))
	coefficient = {}
	std_error = {}
	t_score = {}
	p_value = {}
	# convert data


