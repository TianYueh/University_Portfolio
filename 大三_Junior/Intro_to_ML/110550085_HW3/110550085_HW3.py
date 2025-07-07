# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini_impurity = 1 - np.sum(probabilities**2)
    return gini_impurity

# This function computes the entropy of a label array.
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy_val = -np.sum(probabilities*np.log2(probabilities))
    return entropy_val
    
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    def try_split(self, index, value, dataset):
        l, r = list(), list()
        for row in dataset:
            if row[index] <= value:
                l.append(row)
            else:
                r.append(row)
        return l, r
    
    def select_split(self, dataset):
        tmp_id, tmp_val, tmp_score, tmp_l, tmp_r = 999, 9999, 999 ,None, None
        
        for i in range(len(dataset[0]) - 1):
            thresholds = set([row[i] for row in dataset])
            for t in thresholds:
                if t == max(thresholds):
                    continue
                l, r = self.try_split(i, t, dataset)
                l_labels = [row[-1] for row in l]
                r_labels = [row[-1] for row in r]
                if self.criterion == 'gini':
                    score = (gini(l_labels) * len(l) + gini(r_labels) * len(r)) / (len(l) + len(r))
                else:
                    score = (entropy(l_labels) * len(l) + entropy(r_labels) * len(r)) / (len(l) + len(r))
                if score < tmp_score:
                    tmp_id = i
                    tmp_val = t
                    tmp_score = score
                    tmp_l = l
                    tmp_r = r
        return {'index':tmp_id, 'value':tmp_val, 'left':tmp_l, 'right':tmp_r}
    
    #Split the decision tree recursively
    def split(self, node, depth):
        l = node['left']
        r = node['right']
        
        #Eine Seite ist die Ende nach dem Split
        if not l or not r:
            node['left'] = node['right'] = self.leaf(l + r)
            return 
        
        #Moshi saidai fukasa ni tsuku nara
        if self.max_depth is not None:
            if depth >= self.max_depth:
                node['left'] = self.leaf(l)
                node['right'] = self.leaf(r)
                return 
        
        #links
        l_labels = [row[-1] for row in l]
        if not l_labels.count(0) or not l_labels.count(1):
            node['left'] = self.leaf(l)
        else:
            node['left'] = self.select_split(l)
            if node['left']['index'] == 999:
                node['left'] = self.leaf(l)
            else:
                self.split(node['left'], depth+1)
                
        #rechts
        r_labels = [row[-1] for row in r]
        if not r_labels.count(0) or not r_labels.count(1):
            node['right'] = self.leaf(r)
        else:
            node['right'] = self.select_split(r)
            if node['right']['index'] == 999:
                node['right'] = self.leaf(r)
            else:
                self.split(node['right'], depth+1)
        
    def leaf(self, list):
        labels = [row[-1] for row in list]
        return max(labels, key = labels.count)
    
    def count_feature(self, list, node, X):
        if isinstance(node, dict):
            list[node['index']] += 1
            self.count_feature(list, node['left'], X)
            self.count_feature(list, node['right'], X)
        return list
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        X = list(X)
        y = list(y)
        dataset = [list(X[i]) + [y[i]] for i in range(len(X))]
        self.max_features = int(0)
        self.root = self.select_split(dataset)
        self.split(self.root, 1)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        y_pred = []
        X = list(X)
        for row in X:
            y_pred.append(self.predict_instance(self.root, row))
        y_pred = np.array(y_pred)
        return y_pred
        

    def predict_instance(self, node, row):
        
        if row[node['index']] <= node['value']:
            if isinstance(node['left'], dict):
                return self.predict_instance(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_instance(node['right'], row)
            else:
                return node['right']
            
    def plot_feature_importance_img(self, X):
        #X = list(X)
        attr = [0] * len(X.values.tolist()[0])
        features = self.count_feature(attr, self.root, X.values.tolist())
        names = list(X.columns)

        figure, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, features)  # Adjust color as needed
        ax.invert_yaxis()  # Invert y-axis to have the highest importance at the top

        ax.set_title("Feature Importance", fontsize=16)

        plt.show()
        
        
# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.alphas = []
        self.trees = [DecisionTree(1)] * n_estimators
        
    #Calculate the error of each classifier
    def calculateError(self, tree, X, y):
        index = np.random.choice(len(X), len(X), replace=True, p=self.weights)
        bs_X = [X[i] for i in index]
        bs_X = np.array(bs_X)
        bs_Y = [y[i] for i in index]
        bs_Y = np.array(bs_Y)
        tree.fit(bs_X, bs_Y)
        y_pred = tree.predict(X)
        error = 0
        for i in range(len(X)):
            if y[i] != y_pred[i]:
                error += self.weights[i]
        return error
    
    def weightUpdate(self, alpha, y_data, y_pred):
        f=[]
        y_pred = list(y_pred)
        for i in range(len(y_data)):
            if y_data[i] != y_pred[i]:
                f.append(-1)
            else:
                f.append(1)
        f = np.array(f)
        return self.weights * np.exp(-alpha * f) / sum(self.weights)

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        X = list(X)
        y = list(y)
        self.weights = np.ones(len(X)) * 1 / len(X)
        for i in range(self.n_estimators):
            cError = self.calculateError(self.trees[i], X, y)
            self.alphas.append(np.log((1-cError)/cError)/2)
            y_pred = self.trees[i].predict(X)
            self.weightUpdate(self.alphas[i], y, y_pred)
            
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            t_pred = 0
            for j in range(len(self.trees)):
                pred = self.trees[j].predict_instance(self.trees[j].root, X[i])
                if pred == 0:
                    t_pred -= self.alphas[j]
                else:
                    t_pred += self.alphas[j]
            if t_pred < 0:
                y_pred.append(0)
            else:
                y_pred.append(1)
                
        y_pred = np.array(y_pred)
        return y_pred
    
    
    
# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    df_drop = train_df.drop(["target"], axis=1)
    newtree = DecisionTree(criterion='gini', max_depth = 15)
    newtree.fit(X_train, y_train)
    newtree.plot_feature_importance_img(df_drop)

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=42)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


