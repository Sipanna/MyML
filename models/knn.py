import numpy as np

class KnnClassifier:
    
    def __init__(self, k = 10):
        self.k = k
        
    def predict(self, X, y, X_test):
        y_pred = np.empty(X_test.shape[0])
        for it, xt in enumerate(X_test):
            dist = np.empty(X.shape[0])
            for i, xi in enumerate(X):
                dist[i] = np.linalg.norm(xt - xi)
            idx = dist.argsort()[:self.k]
            ny = y[idx]
            classes = np.unique(ny, return_counts=True)
            ind = np.argmax(classes[1])
            y_pred[it] = classes[0][ind]
        return y_pred
    
    
class KnnHyperplanesClassifier:
    def __init__(self, k = 3, bucket_size = 16):
        self.k = k
        self.bucket_size = bucket_size
        
    def fit(self, X, y):
        self.bucket_size = self.bucket_size
        self.X = X
        self.y = y
        self.hyperplanes = self.generate_hyperplanes()
        self.hash_table = self.locality_sensitive_hash()
        print("len hash", len(self.hash_table))
        print("hash", self.hash_table.keys())
        
        
    def generate_hyperplanes(self):
        n = self.X.shape[0] # кол-вл наблюдений
        m = self.X.shape[1] # кол-во признаков
        b = n // self.bucket_size
        print("b", b)
        h = int(np.log2(b))
        print("h", h)
        H = np.random.normal(size=(h, m))
        return H
        
    def hamming_hash(self, X):
        h = len(self.hyperplanes)
        hash_key = (X @ self.hyperplanes.T) >= 0
        
        dec_vals = np.array([2**i for i in range(h)])
        hash_key = hash_key @ dec_vals
        return hash_key
    
    def locality_sensitive_hash(self):
        hash_vals = self.hamming_hash(self.X)
        hash_table = {}
        for i, v in enumerate(hash_vals):
            if v not in hash_table:
                hash_table[v] = set()
            hash_table[v].add(i)
        return hash_table
    
    
    def predict(self, X_test):
        y_pred = np.empty(X_test.shape[0])
        hash_test = self.hamming_hash(X_test)
        
        for i, query in enumerate(X_test):
            query_hash = hash_test[i]
            candidates = set()
            if query_hash in self.hash_table:
                candidates = self.hash_table[query_hash]
            candidates = np.array([self.X[i] for i in candidates])
            dists = np.sqrt(np.sum((candidates - query)**2, axis = 1))
            inds = np.argsort(dists)
            inds_k = inds[:self.k]
            #print("X", self.X[inds_k])
            #print("dist", dists[inds_k])
            #print("y", self.y[inds_k])
            ny = y[inds_k]
            classes = np.unique(ny, return_counts=True)
            ind = np.argmax(classes[1])
            y_pred[i] = classes[0][ind]
        return y_pred

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples = 200, n_features = 2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

print("X shape", X.shape)

lsh = KnnHyperplanesClassifier(k = 5)
lsh.fit(X_train, y_train)
y_pred = lsh.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("score", score)