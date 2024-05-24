import numpy as np

class DecisionTree:
    def __init__(self, max_depth, metric='gini'):
        self.max_depth = max_depth
        self.metric = metric
        self.root = None
        
    def _impurity(self, y):
        if self.metric == 'gini':
            return self._gini(y)
        elif self.metric == 'entropy':
            return self._entropy(y)
        elif self.metric == 'error':
            return self._error(y)
        else:
            raise ValueError('Unknown metric')
        
    def _error(self, y):
        if len(y) == 0:
            return 0
        cnt= np.count_nonzero(y == 1)
        p = cnt/len(y)
        
        err = 1 - max(p, 1-p)
        
        return err
        
    def _entropy(self, y):
        if len(y) == 0:
            return 0
        cnt = np.count_nonzero(y == 1)
        p = cnt/len(y)
        if p == 0 or p == 1:
            return 0
        entropy =  p * np.log2(p) + (1-p) * np.log2(1-p)
        
        return -entropy
        
    def _gini(self, y):
        # print(y, "gini")
        if len(y) == 0:
            return 0
        cnt = np.count_nonzero(y == 1)
        p = cnt/len(y)
        
        gini = 2 * p * (1-p)
        
        return gini

    def find_best_split(self, X, y):
        # some features have binary values, some are ternary
        best_feature = None
        best_val = None
        best_impurity = float('inf')
        
        # print(X.shape, y.shape)
        values = np.unique(y)
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for val in values:
                left_idx = X[:, feature] < val
                right_idx = X[:, feature] >= val
                left_y = y[left_idx]
                right_y = y[right_idx]
                impurity = len(left_y)/len(y) * self._impurity(left_y) + len(right_y)/len(y) * self._impurity(right_y)
                if impurity < best_impurity:
                    best_feature = feature
                    best_val = val
                    best_impurity = impurity
        
        if best_feature is None:
            print(X.shape,y, "none")
        # print('Best impurity:', best_impurity)
        # print('Best feature:', best_feature)
        # print('Best value:', best_val)
        return best_feature, best_val    
                
    def _create_tree(self, X, y, depth):
        if depth == self.max_depth:
            value = np.median(y)
            return self.TreeNode(value=value)
        if len(np.unique(y)) == 1:
            return self.TreeNode(value=y[0])
        # print('Creating tree at depth:', depth)
        
        split_feature, split_val = self.find_best_split(X, y)
        # to solve: what if one of the splits is empty?
        left_idx = X[:, split_feature] < split_val
        right_idx = X[:, split_feature] >= split_val
        # print('Left:', len(y[left_idx]))
        # print('Right:', len(y[right_idx]))
        if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
            value = np.median(y)
            return self.TreeNode(value=value)
        left = self._create_tree(X[left_idx], y[left_idx], depth+1)
        right = self._create_tree(X[right_idx], y[right_idx], depth+1)
        
        return self.TreeNode(feature=split_feature, split_val=split_val, left=left, right=right)
        
    def fit(self, X, y):
        self.root = self._create_tree(X.values, y.values, 0)
        
    def _predict(self, x, node):
        # Implementation of recursive prediction
        if node.value is not None:
            return node.value
        if x[node.feature] < node.split_val:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)
        
    def predict(self, X):
        return [self._predict(x, self.root) for x in X.values] 
    
    def best_feature(self):
        return self.root.feature
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)*100  
    
    class TreeNode:
        def __init__(self, feature=None, split_val=None, left=None, right=None, value=None):
            self.feature = feature
            self.split_val = split_val
            self.left = left
            self.right = right
            self.value = value