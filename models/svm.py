import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

#Через градиентный спуск
class LinearSVM:
    
    def __init__(self, C = 0.05, learning_rate = 0.01, iteartions = 3000):
        self.learning_rate = learning_rate
        self.iterations = iteartions
        self.C = C
        self.track_w = []
        self.track_loss = []
        self.track_gr = []
        
    def fit(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]),X])
        y[y==0] = -1
        w = np.random.rand(X.shape[1]) 
        n = X.shape[0] #количество наблюдений
                
        for i in range(self.iterations):           
            grad = self.C * w
            for j in range(n):
                margin  = y[j] * (X[j] @ w)
                if margin < 1: # неверно
                    grad -= y[j]*X[j]/n
            w = w - self.learning_rate * grad
        self.bias, self.weights = w[0], w[1:]

    
    def predict(self, X_test):
        y_pred = X_test @ self.weights + self.bias
        return [1 if yi>=1 else -1 for yi in y_pred]

    
#Simplified SMO    
class KernelSVM_SSMO:
    
    def __init__(self, C = 0.5, kernel='linear', degree = 2, max_passes = 30, tol = 0.01, gamma = 1):
        self.max_passes = max_passes
        self.C = C
        self.tol = tol
        self.degree = degree
        self.gamma = gamma
        self.K = {'poly'  : lambda x,y: np.dot(x, y.T)**degree,
         'rbf': lambda x,y: np.exp(-gamma*np.sum((y-x[:,np.newaxis])**2,axis=-1)),
         'linear': lambda x,y: np.dot(x, y.T)}[kernel]
        
        
    def error(self, ind, lambdas, b):
        m = self.X.shape[0]
        f = 0
        for i in range(m):
            f += lambdas[i]*self.y[i]*self.K(self.X[i], self.X[ind])
        return f - b - self.y[ind]
        
        
    def find_lambdas(self, X, y):
        m = X.shape[0]
        lambdas = np.zeros(m)
        eps = 1E-5
        b = 0
        passes = 0
        while(passes < self.max_passes):
            num_changed_lambdas = 0
            for i in range(m):
                E1 = self.error(i, lambdas, b)
                if (E1*y[i] < -self.tol and lambdas[i] < self.C) or (E1*y[i] > self.tol and lambdas[i] > 0):
                    j = i
                    while j == i:
                        j = np.random.randint(0, len(lambdas))
                    E2 = self.error(j, lambdas, b)
                    lambda_i = lambdas[i]
                    lambda_j = lambdas[j]
                    if y[i] == y[j]:
                        L = max(0, lambda_i + lambda_j - self.C)
                        H = min(self.C, lambda_i + lambda_j)
                    else:
                        L = max(0, lambda_j - lambda_i)
                        H = min(self.C, self.C - lambda_i + lambda_j)
                    if L == H:
                        continue
                    
                    K11 = self.K(X[i], X[i])
                    K22 = self.K(X[j], X[j])
                    K12 = self.K(X[i], X[j])
                    eta = K11 + K22 - 2*K12
                    if eta <= 0:
                        continue
                    lambdas[j] = lambda_j  + y[j]*(E1 - E2)/eta
                    self.K(X[i], X[i])
                    if lambdas[j] > H:
                        lambdas[j] = H
                    elif lambdas[j] < L:
                        lambdas[j] = L
                    if abs(lambda_j - lambdas[j]) < eps:
                        continue
                        
                    lambdas[i] = lambda_i + y[i]*y[j]* (lambda_j - lambdas[j])
                    
                    b1 = b + y[i]*K11*(lambdas[i] - lambda_i) + y[j]*K12*(lambdas[j] - lambda_j) + E1
                    b2 = b + y[i]*K12*(lambdas[i] - lambda_i) + y[j]*K22*(lambdas[j] - lambda_j) + E2
                    if 0 < lambdas[i] < self.C:
                        b = b1
                    elif  0 < lambdas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2)/2.0
                    num_changed_lambdas += 1
            if num_changed_lambdas == 0:
                passes += 1
            else:
                passes = 0
        return lambdas, b        
                
            
    def fit(self, X, y):
        y[y==0] = -1
        self.y = y
        self.X = X
        self.lambdas, self.b = self.find_lambdas(X, y)
        
    def get_params(self):
        w =  np.sum(self.X.T *(self.y * self.lambdas), axis = 0)
        return self.b, w    

    def predict(self, X_test):
        y_pred = np.sum(self.K(X_test, self.X) * self.y * self.lambdas, axis=1) - self.b
        print (y_pred)
        return [1 if yi>=0 else -1 for yi in y_pred]
    
    
class KernelSVM_SMO:
    
    def __init__(self, C = 0.05, kernel='linear', degree = 2,  tol = 0.01, gamma = 1):
        self.C = C
        self.tol = tol
        self.degree = degree
        self.gamma = gamma
        self.K = {'poly'  : lambda x,y: np.dot(x, y.T)**degree,
         'rbf': lambda x,y:np.exp(-gamma*np.sum((y-x[:, np.newaxis])**2, axis = -1)),
         'linear': lambda x,y: np.dot(x, y.T)}[kernel]
        
        
    def f(self, ind):
        return np.sum([self.lambdas[j] * self.y[j]* self.K(self.X[j], self.X[ind])
                           for j in range(self.m)]) - self.b
        
    def get_error(self, ind):
        if 0 < self.lambdas[ind] < self.C:
            return self.errors[ind]
        else:
            return self.f(ind) - self.y[ind]
        
    
        
    def take_step(self, i1, i2):
        print("i1, i2", (i1, i2))
        if i1 == i2:
            return False
        
        E1 = self.get_error(i1)
        y1 = self.y[i1]
        X1 = self.X[i1]
        print("y1, y2", (y1, self.y2))
        lambda1_old = self.lambdas[i1]
        lambda2_old = self.lambdas[i2]
        if y1 == self.y2:
            L = max(0, lambda1_old + lambda2_old - self.C)
            H = min(self.C, lambda1_old + lambda2_old)
        else:
            L = max(0, lambda2_old - lambda1_old)
            H = min(self.C, self.C - lambda1_old + lambda2_old)
        if L == H:
            return False
                 
        K11 = self.K(X1.reshape(1, -1), X1.reshape(1, -1))
        K22 = self.K(self.X2.reshape(1, -1), self.X2.reshape(1, -1))
        K12 = self.K(X1.reshape(1, -1), self.X2.reshape(1, -1))
        
        print("K11, K12, K22", (K11, K12, K22))            
        eta = K11 + K22 - 2*K12
        if eta <= 0:
            return False
        
        self.lambdas[i2] = lambda2_old  + self.y2*(E1 - self.E2)/eta
        print("Lambda2", self.lambdas[i2])
        if self.lambdas[i2] > H:
            self.lambdas[i2] = H
        elif self.lambdas[i2] < L:
            self.lambdas[i2] = L
        print("Lambda2 clipped", self.lambdas[i2])    
        if abs(self.lambdas[i2] - lambda2_old) < self.eps * (self.lambdas[i2] + lambda2_old + self.eps):
            return False
                        
        self.lambdas[i1] = lambda1_old + y1*self.y2* (lambda2_old - self.lambdas[i2])
        print("Lambda1", self.lambdas[i1])            
                    
        b1 = self.b + y1*K11*(self.lambdas[i1] - lambda1_old) + self.y2*K12*(self.lambdas[i2] - lambda2_old) + E1
        b2 = self.b + y1*K12*(self.lambdas[i1] - lambda1_old) + self.y2*K22*(self.lambdas[i2] - lambda2_old) + self.E2
        if 0 < self.lambdas[i1] < self.C:
            new_b = b1
        elif  0 < self.lambdas[i2] < self.C:
            new_b = b2
        else:
            new_b = (b1 + b2)/2.0
        
        delta_b = new_b - self.b
        self.b = new_b
        
        print("newf", (self.f(i1), self.f(i2)))
        delta1 = y1 * (self.lambdas[i1] - lambda1_old)
        delta2 = self.y2 * (self.lambdas[i2] - lambda2_old)
        
        for i in range(self.m):
            if 0 < self.lambdas[i] < self.C:
                self.errors[i] += delta1 * self.K(X1.reshape(1, -1), self.X[i].reshape(1, -1)) + delta2 * self.K(self.X2.reshape(1, -1), self.X[i].reshape(1, -1)) - delta_b
        
             
        self.errors[i1] = 0
        self.errors[i2] = 0
        self.track_f.append(self.decision_function(self.X))
        return True
    
    def second_heuristic(self, non_bound_indices):
        i1 = -1
        if len(non_bound_indices) > 1:
            max = 0
            for j in non_bound_indices:
                E1 = self.errors[j] - self.y[j]
                step = abs(E1 - self.E2) 
                if step > max:
                    max = step
                    i1 = j
        return i1
    
    
    def examine_example(self, i2):
        self.y2 = self.y[i2]
        lambda2 = self.lambdas[i2]
        self.X2 = self.X[i2]
        self.E2 = self.get_error(i2)

        r2 = self.E2 * self.y2

        if not ((r2 < -self.tol and lambda2 < self.C) or (r2 > self.tol and lambda2 > 0)):
            # Условия ККТ выполняются, нужен другой индекс
            return 0

        #Вторая эвристика 1 - выбор множителя с максимальной ошибкой
        non_bound_idx = list(self.get_non_bound_indexes())
        i1 = self.second_heuristic(non_bound_idx)

        if i1 >= 0 and self.take_step(i1, i2):
            return 1

        # Вторая эвристика 2 - Если нет таких множителей, то цикл из рандомной точки по всем не граничным lambda
        if len(non_bound_idx) > 0:
            rand_i = np.random.randint(len(non_bound_idx))
            for i1 in non_bound_idx[rand_i:] + non_bound_idx[:rand_i]:
                if self.take_step(i1, i2):
                    return 1

        # Вторая эвристика 3 - Цикл из рандомной точки по всем lambda
        rand_i = np.random.randint(self.m)
        all_indices = list(range(self.m))
        for i1 in all_indices[rand_i:] + all_indices[:rand_i]:
            if self.take_step(i1, i2):
                return 1

        return 0
    
    def get_non_bound_indexes(self):
        return np.where(np.logical_and(self.lambdas> 0,
                                       self.lambdas < self.C))[0]
        
    def first_heuristic(self):
        num_changed = 0
        non_bound_idx = self.get_non_bound_indexes()

        for i in non_bound_idx:
            num_changed += self.examine_example(i)
        return num_changed
    
    
    

    def main_routine(self):
        num_changed = 0
        examine_all = True

        while num_changed > 0 or examine_all:
            num_changed = 0

            if examine_all:
                for i in range(self.m):
                    num_changed += self.examine_example(i)
            else:
                num_changed += self.first_heuristic()

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
                
            
    def fit(self, X, y):
        y[y==0] = -1
        self.y = y
        self.X = X
        self.m = X.shape[0]
        self.errors = np.zeros(self.m)
        self.lambdas = np.zeros(self.m)
        self.eps = 1E-3
        self.b = 0
        self.track_f = []
        self.main_routine()
        
        
    def get_params(self):
        w =  np.sum(self.X.T *(self.y * self.lambdas), axis = 1)
        return self.b, w, self.lambdas, self.track_f
    
    def decision_function(self, X_test):
        res = (np.sum(self.K(X_test, self.X) * self.y * self.lambdas, axis=1) - self.b).reshape(X_test.shape[0])
        return  res

    def predict(self, X_test):
        y_pred = (np.sum(self.K(X_test, self.X) * self.y * self.lambdas, axis=1) - self.b).reshape(X_test.shape[0])
        print('Y_pred', y_pred)
        return [1 if yi>=0 else -1 for yi in y_pred]