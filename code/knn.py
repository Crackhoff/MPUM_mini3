import numpy as np

class KNN:
    def __init__(self, n_neighbors, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        
    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.linalg.norm(x1 - x2)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'l_inf':
            return np.max(np.abs(x1-x2))
        elif self.metric == 'adam':
            # if equal, return 0, else return 1
            return np.sum(np.abs(x1 - x2) > 0)
        else:
            raise ValueError('Unknown metric')
        
    def fit(self, X, y):
        X = X.values
        y = y.values
        self.X = X
        self.y = y
                    
    def _predict(self, x):
        neighbors = []
        for i in range(len(self.X)):
            distance = self._distance(x, self.X[i])
            neighbors.append((distance, self.y[i]))

        neighbors = sorted(neighbors, key=lambda x: x[0])
        neighbors = neighbors[:self.n_neighbors]
        classes = [x[1] for x in neighbors]
        return max(set(classes), key=classes.count)
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X.values])
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)*100