import numpy as np

class SVM:
    def __init__(self, kernel, max_iter=10000,learning_rate=0.001,):
        self.learning_rate = learning_rate
        self.kernel = kernel # function
        self.max_iter = max_iter
        
    def _predict(self, x):
        # Compute prediction
        return np.dot(x, self.w)
    
    def fit(self, X, y):
       # Fit the model using SVM and gradient descent
        self.X = X.values
        self.y = y.values
        self.w = np.zeros(X.shape[1])
        for _ in range(self.max_iter):
            if _ % 1000 == 0:
                print(f'Epoch: {_}')
            # gradient descent
            for i in range(len(self.X)):
                if self.y[i] * self._predict(self.X[i]) < 1:
                    self.w = self.w + self.learning_rate * ((self.X[i] * self.y[i]) + (-2 * (1/self.max_iter) * self.w))
                else:
                    self.w = self.w + self.learning_rate * (-2 * (1/self.max_iter) * self.w)
                                
        return self.w
    
    def predict(self, X):
        X = X.values
        return np.array([1 if self._predict(x) > 0 else -1 for x in X])
    
    def score(self, X, y):
        # Compute accuracy
        return np.mean(self.predict(X) == y)*100
    
    def best_hyperplane(self):
        # Return the hyperplane parameters
        return self.w
    
class KernelFactory:
    @staticmethod
    def create_kernel(name, kernel_params=None):
        if name == 'gaussian':
            def gaussian_kernel(x1, x2):
                return np.exp(-np.linalg.norm(x1 - x2)**2)
            return gaussian_kernel
        elif name == 'sigmoid':
            def sigmoid_kernel(x1, x2):
                return np.tanh(np.dot(x1, x2) + kernel_params['coef'])
            return sigmoid_kernel
        elif name == 'linear':
            def linear_kernel(x1, x2):
                return np.dot(x1, x2)
            return linear_kernel
        elif name == 'polynomial':
            def polynomial_kernel(x1, x2):
                return (np.dot(x1, x2) + kernel_params['coef'])**kernel_params['degree']
            return polynomial_kernel
        elif name == 'rbf':
            def rbf_kernel(x1, x2):
                return np.exp(-kernel_params['gamma'] * np.linalg.norm(x1 - x2)**2)
            return rbf_kernel
        else:
            raise ValueError('Unknown Kernel')

    
class SMO:
    def __init__(self, kernel, C, eps):
        self.kernel = kernel
        self.C = C
        self.eps = eps
        self.target = None
        self.alpha = None
        self.b = 0
        
    def _predict(self, x):
        return np.dot(self.alpha * self.target, [self.kernel(x, xi) for xi in self.X]) + self.b
    
    def objective_function(self, alphaj, i,j):
        alphas = np.copy(self.alpha)
        alphas[j] = alphaj
        W = np.sum(alphas) - 0.5 * np.sum(alphas[i] * alphas[j] * self.target[i] * self.target[j] * self.kernel(self.X[i], self.X[j]))
        return W
        
    def take_step(self,i,j):
        if i == j:
            return 0
        alphai = self.alpha[i]
        alphaj = self.alpha[j]
        yi = self.y[i]
        yj = self.y[j]
        Ei = self.errors[i]
        Ej = self.errors[j]
        s = yi * yj
        a1 = 0
        a2 = 0
        L =0
        H = 0
        if yi != yj:
            L = max(0, alphaj - alphai)
            H = min(self.C, self.C + alphaj - alphai)
        else:
            L = max(0, alphaj + alphai - self.C)
            H = min(self.C, alphaj + alphai)
        if L == H:
            return 0
        kii = self.kernel(self.X[i], self.X[i])
        kij = self.kernel(self.X[i], self.X[j])
        kjj = self.kernel(self.X[j], self.X[j])
        eta = 2 * kij - kii - kjj
        if eta < 0:
            a2 = alphaj - yj * (Ei - Ej) / eta
            if a2 > H:
                a2 = H
            elif a2 < L:
                a2 = L
        else:
            Lobj = self.objective_function(L, i, j)
            Hobj = self.objective_function(H, i, j)
            if Lobj > Hobj + self.eps:
                a2 = L
            elif Lobj < Hobj - self.eps:
                a2 = H
            else:
                a2 = alphaj
        if abs(a2 - alphaj) < self.eps * (a2 + alphaj + self.eps):
            return 0
        
        a1 = alphai + s * (alphaj - a2)
        
        b1 = self.b - Ei - yi * (a1 - alphai) * kii - yj * (a2 - alphaj) * kij
        b2 = self.b - Ej - yi * (a1 - alphai) * kij - yj * (a2 - alphaj) * kjj
        if 0 < a1 < self.C:
            self.b = b1
        elif 0 < a2 < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
            
        self.alpha[i] = a1
            
        self.errors[i] = self._predict(self.X[i]) - self.y[i]
        self.errors[j] = self._predict(self.X[j]) - self.y[j]
        
        self.target = self.y * self.alpha
        return 1    
        
    
    def examine_example(self, i):
        y1 = self.y[i]
        alphai = self.alpha[i]
        Ei = self.errors[i] - y1
        ri = Ei * y1
        if ri < -self.eps and alphai < self.C or ri > self.eps and alphai > 0:
            if len(self.alpha[(self.alpha != 0) & (self.alpha != self.C)]) > 1:
                if self.errors[i] > 0:
                    j = np.argmin(self.errors)
                else:
                    j = np.argmax(self.errors)
                if self.take_step(i, j):
                    return 1
            for j in np.roll(np.where((self.alpha != 0) & (self.alpha != self.C))[0], np.random.choice(range(len(self.alpha)))):
                if self.take_step(i, j):
                    return 1
            for j in np.roll(range(len(self.alpha)), np.random.choice(range(len(self.alpha)))):
                if self.take_step(i, j):
                    return 1
        return 0
    
    def fit(self, X, y):
        X = X.values
        y = y.values
        self.alpha = np.zeros(len(X))
        self.errors = np.zeros(len(X))
        self.target = y
        self.b = 0
        self.X = X
        self.y = y
        changed = 0
        examine_all = True
        while changed > 0 or examine_all:
            changed = 0
            if examine_all:
                for i in range(len(X)):
                    changed += self.examine_example(i)
            else:
                for i in range(len(X)):
                    if 0 < self.alpha[i] < self.C:
                        changed += self.examine_example(i)
            if examine_all:
                examine_all = False
            elif changed == 0:
                examine_all = True
        
    def fit_test(self, X, y,X_test,y_test):
        X = X.values
        y = y.values
        self.alpha = np.zeros(len(X))
        self.errors = np.zeros(len(X))
        self.target = y
        self.b = 0
        self.X = X
        self.y = y
        changed = 0
        examine_all = True
        cntr = 0
        j =0
        while changed > 0 or examine_all:
            j+=1
            if j%10 == 0:
                print(j)
            changed = 0
            if examine_all:
                for i in range(len(X)):
                    changed += self.examine_example(i)
            else:
                for i in range(len(X)):
                    if 0 < self.alpha[i] < self.C:
                        changed += self.examine_example(i)
            if examine_all:
                examine_all = False
            elif changed == 0:
                examine_all = True
                print(self.score(X_test,y_test))
    
    def predict(self, X):
        X = X.values
        return np.array([1 if self._predict(x) > 0 else -1 for x in X])
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)*100