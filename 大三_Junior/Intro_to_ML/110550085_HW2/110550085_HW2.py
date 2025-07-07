# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.intercept = 0

        for _ in range(self.iteration):
            # Calculate the logistic function (sigmoid)
            z = np.dot(X, self.weights) + self.intercept
            sigmoid = self.sigmoid(z)

            # Calculate the partial derivatives of the cross-entropy loss w.r.t. weights and intercept
            d_weights = np.dot(X.T, (sigmoid - y)) / len(y)
            d_intercept = np.mean(sigmoid - y)

            # Update weights and intercept using the partial derivatives
            self.weights -= self.learning_rate * d_weights
            self.intercept -= self.learning_rate * d_intercept
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        z = np.dot(X, self.weights) + self.intercept
        sigmoid = self.sigmoid(z)

        # Assign class labels based on a threshold (e.g., 0.5)
        y_pred = (sigmoid >= 0.5).astype(int)
        return y_pred

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        # Separate the data into two classes (0 and 1)
        X0 = X[y == 0]
        X1 = X[y == 1]

        # Calculate class means
        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)

        # Calculate within-class scatter matrix (sw)
        sw0 = np.dot((X0 - self.m0).T, X0 - self.m0)
        sw1 = np.dot((X1 - self.m1).T, X1 - self.m1)
        self.sw = sw0 + sw1

        # Calculate between-class scatter matrix (sb)
        mean_diff = self.m0 - self.m1
        self.sb = np.outer(mean_diff, mean_diff)

        # Calculate the Fisher's Linear Discriminant (FLD) vector
        sw_inv=np.linalg.inv(self.sw)
        tmp=np.matmul(sw_inv, self.m0-self.m1)
        self.w=tmp/np.linalg.norm(tmp)
        
        '''
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(self.sw).dot(self.sb))
        # Select the eigenvector with the largest eigenvalue
        self.w = eigenvectors[:, np.argmax(eigenvalues)]
        # Normalize the FLD vector to have unit length
        self.w = self.w / np.linalg.norm(self.w)
        '''
        
        # Calculate the slope for the projection line
        self.slope = self.w[1] / self.w[0]


    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        # Project data onto the FLD direction
        projection = np.dot(X, self.w)

        # Project class means
        projected_m0 = np.dot(self.m0, self.w)
        projected_m1 = np.dot(self.m1, self.w)

        # Classify based on the distance from projected class means
        y_pred = [0 if abs(p - projected_m0) < abs(p - projected_m1) else 1 for p in projection]

        return y_pred

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    
    def plot_projection(self, X, y):
        # Split data into two classes
        X0 = X[y == 0]
        X1 = X[y == 1]
        
        
        x0_project=np.zeros((len(X0), 2))
        x1_project=np.zeros((len(X1), 2))
        
        for i in range(len(X0)):
            length=np.dot(X0[i], self.w)
            x0_project[i]=self.w*length
            
        for i in range(len(X1)):
            length=np.dot(X1[i], self.w)
            x1_project[i]=self.w*length
        
        x0, x1, y0, y1 = [], [], [], []
        
        for i in range(len(X0)):
            x0.append(X0[i][0])
            x0.append(x0_project[i][0])
            y0.append(X0[i][1])
            y0.append(x0_project[i][1])
        for i in range(len(X1)):
            x1.append(X1[i][0])
            x1.append(x1_project[i][0])
            y1.append(X1[i][1])
            y1.append(x1_project[i][1])
            
        plt.scatter(x0, y0, color='red', s=3)
        plt.scatter(x1, y1, color='blue', s=3)
        
        for i in range(len(X0)):
            xp = [x0_project[i][0], X0[i][0]]
            yp = [x0_project[i][1], X0[i][1]]
            plt.plot(xp, yp, color='slategrey', alpha=0.4, linewidth=0.5)
        
        for i in range(len(X1)):
            xp = [x1_project[i][0], X1[i][0]]
            yp = [x1_project[i][1], X1[i][1]]
            plt.plot(xp, yp, color='slategrey', alpha=0.4, linewidth=0.5)
        
        # Get the two attributes for the x and y axis
        x0_axis_attribute = x0_project[:, 0]
        x1_axis_attribute = x1_project[:, 0]
        #print(x0_axis_attribute)
        # Project the line with the given slope
        x_line = np.linspace(min(x0_axis_attribute-10), max(x1_axis_attribute+10), 100)
        y_line = self.slope * x_line 
        
        '''
        # Plot the projection line
        plt.scatter(X0[:, 0], X0[:, 1], label='Class 0', marker='o', color='blue')
        plt.scatter(X1[:, 0], X1[:, 1], label='Class 1', marker='o', color='red')
        '''
        plt.plot(x_line, y_line, label='Projection Line', color='slategrey')
        
        # Set labels and legend
        plt.xlabel('Age')
        plt.ylabel('thalach')
        plt.title(f'Projection Line: w={self.slope:.6f}, b=0', loc='center', fontsize=12)

        #plt.legend()

        # Show the plot
        plt.show()


# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.0001, iteration=100000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"
    #y_pred_arr=np.array(y_pred)
    #FLD.plot_projection(X_test, y_pred_arr)

