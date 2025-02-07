import numpy as np

class LinearRegression:
        
    def fit(self, X, y):
        if X.shape[1] > 1 :
            X = np.column_stack([np.ones(X.shape[0]),X])
            w = (np.linalg.inv(X.T @ X) @ X.T) @ y
            self.bias = w[0]
            self.weights = w[1:]
        else:
            X = X[:, 0]
            k = (X.mean() * y.mean() - (X*y).mean())/ (X.mean()**2 - (X*X).mean())
            b = y.mean() - k * X.mean()
            self.weights = np.array([k])
            self.bias = b
            print("k", k)
            print("b", b)
    
    def predict(self, X_test):
        return X_test @ self.weights + self.bias
    
    
    
class RidgeRegression:
    
    def __init__(self, alpha = 0.01):
        self.alpha = alpha
        
    def fit(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]),X])
        Xt_X = X.T @ X
        w = (np.linalg.inv(Xt_X + self.alpha * np.identity(Xt_X.shape[0])) @ X.T) @ y
        self.bias = w[0]
        self.weights = w[1:]
    
    def predict(self, X_test):
        return X_test @ self.weights + self.bias
    
  
    
class GDRegression:
    
    def __init__(self, learning_rate = 0.01, iterations = 200):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def mse(self, X, y, w):
        n = len(y)
        err = (1.0/n) * (np.linalg.norm(X @ w - y)**2)
        return err
    
        
    def fit(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]),X])
        w = np.random.rand(X.shape[1]) 
        n = X.shape[0] #количество наблюдений
        
        err_track = np.zeros((self.iterations, 0)) 
        for i in range(self.iterations):
            grad = (2.0/n) * X.T @ (X @ w - y) 
            w = w - self.learning_rate * grad
            err_track[i] = self.mse(X, y, w)
        self.bias, self.weights = w[0], w[1:]
        return err_track
    
    def predict(self, X_test):
        return X_test @ self.weights + self.bias
    
    
class SGDRegression:
    
    def __init__(self, learning_rate = 0.01, iterations = 100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def mse(self, X, y, w):
        n = len(y)
        err = (1.0/n) * (np.linalg.norm(X @ w - y)**2)
        return err
    
        
    def fit(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]),X])
        w = np.random.rand(X.shape[1]) 
        
        err_track = np.zeros((self.iterations, 0)) 
        for i in range(self.iterations):
            j = np.random.randint(len(X))
            grad = 2.0 * X[j].T * (X[j] @ w - y[j]) 
            w = w - self.learning_rate * grad
            err_track[i] = self.mse(X, y, w)
        self.bias, self.weights = w[0], w[1:]
        return err_track
    
    def fit_batch(self, X, y, b = 20):
        X = np.column_stack([np.ones(X.shape[0]),X])
        w = np.random.rand(X.shape[1]) 
        n = len(y)
        err_track = np.zeros((self.iterations, 0)) 
        for i in range(self.iterations):
            #перемешиваем
            Xn = np.column_stack([X,y])
            np.random.shuffle(Xn)
            yn = Xn[:,-1]
            Xn = Xn[:, :-1]
            
            for i in range(0, n, b):
                Xb = Xn[i:i+b]
                yb = yn[i:i+b]
                grad = (2.0/b) * Xb.T @ (Xb @ w - yb) 
            w = w - self.learning_rate * grad
            err_track[i] = self.mse(X, y, w)
        self.bias, self.weights = w[0], w[1:]
        return err_track
    
    def predict(self, X_test):
        return X_test @ self.weights + self.bias
    
    
class LogisticRegression:
    
    def __init__(self, learning_rate = 0.01, iteartions = 200):
        self.learning_rate = learning_rate
        self.iterations = iteartions
        
    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))
    
    def fit(self, X,  y):
        X = np.column_stack([np.ones(X.shape[0]),X])
        w = np.random.rand(X.shape[1]) 
        n = X.shape[0] #количество наблюдений
                
        for i in range(self.iterations):
            y_pred  = X @ w
            j = np.random.randint(len(X))
            grad = (1.0/n) * (X.T @ (self.sigmoid(y_pred) - y))
            w = w - self.learning_rate * grad
        self.bias, self.weights = w[0], w[1:]

    def get_params(self):
        return self.bias, self.weights
    
    def predict(self, X_test):
        y_pred = X_test @ self.weights + self.bias
        proba = self.sigmoid(y_pred)
        return [1 if pr > 0.5 else 0 for pr in proba]
    
    def predict_proba(self, X_test):
        y_pred = X_test @ self.weights + self.bias
        return self.sigmoid(y_pred)


