import numpy as np
import pickle


class ALS_recommend_no_spark(object):
    def __init__(self):
        filename = 'ALSItem'
        infile = open(filename, 'rb')
        self.ALSItem = pickle.load(infile)
        infile.close()
        filename = 'ALSUser'
        infile = open(filename, 'rb')
        self.ALSUser = pickle.load(infile)
        infile.close()


    def predict(self, X, feature_names):
        who = [X[0].astype(int).item()]
        what = [X[1].astype(int).item()]
        return np.asarray([np.array(["score"]), self.ALSUser[who, :].dot(self.ALSItem[what, :].T)])







