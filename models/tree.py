import numpy as np

class DecisionTree:
    
    def __init__(self, max_depth = 5):
        self.max_depth = max_depth
        
    def H(self, y):
        classes = np.unique(y, return_counts=True)
        h = 0
        k = len(y)
        for i in range(len(classes[1])):
            p = classes[1][i]/k
            h -= p*np.log2(p)
        return h

    def IG(self, y, yl, yr):
        r = len(y)
        rl = len(yl)
        rr = len(yr)
        Ig = self.H(y) - rl*self.H(yl)/r - rr*self.H(yr)/r
        return Ig
    
    def get_best_partition(self, X, y):
        best_f = 0
        best_value = 0
        best_IG = 0 
        for f in range(X.shape[1]):    
            values = np.unique(X[:, f])
            for value in values:
                yl = y[X[:,f] < value]
                yr = y[X[:, f] >= value]
                Ig = self.IG(y, yl=yl, yr = yr)
                if Ig > best_IG:
                    best_IG = Ig
                    best_f = f
                    best_value = value
        return best_f, best_value
    
    def get_result(self, y):
        h = self.H(y)
        classes = np.unique(y, return_counts=True)
        ind = np.argmax(classes[1])
        cl = classes[0][ind]
        return (cl, h)
    
    def get_tree(self, X, y, depth):
        f, value = self.get_best_partition(X, y)
        if depth > 1:            
            if len(y[X[:,f] < value]) > 0:
                left_tree = self.get_tree(X[X[:,f] < value], y[X[:,f] < value], depth - 1)
            else:
                left_tree = {}
            if len(y[X[:,f] >= value]) > 0: 
                right_tree = self.get_tree(X[X[:,f] >= value], y[X[:,f] >= value], depth - 1)
            else:
                right_tree = {}
            if right_tree=={} or left_tree=={}:
                return {"f":-1, "class": self.get_result(y)[0], "p" : self.get_result(y)[1] }
            return {"f" : f, "value" : value, "left" :  left_tree, 
                    "right" : right_tree}
        else:
            return {"f" : -1, "class": self.get_result(y)[0], "p" : self.get_result(y)[1]}
        
    def fit(self, X, y):
        self.tree = self.get_tree(X, y, depth = self.max_depth)
        
    def predict_item(self, tree,  x):
        if tree['f'] != -1:
            ind = tree['f']
            if x[ind] < tree['value']:
                return self.predict_item(tree['left'], x)
            else:
                return self.predict_item(tree['right'], x)
        else:
            return tree['class']

    def predict(self, X):
        y_pred = []
        for x in X:
            yi = self.predict_item(self.tree, x)
            y_pred.append(yi)
        return y_pred