# You are not allowed to import any additional packages/libraries.
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        
    # This function computes the closed-form solution of linear regression.
    def closed_form_fit(self, X, y):
        # Compute closed-form solution.
        # Save the weights and intercept to self.closed_form_weights and self.closed_form_intercept
        X = np.c_[np.ones((8000, 1)), X]
        X_T = np.transpose(X)
        X_T_X = np.dot(X_T, X)
        X_T_X_inv = np.linalg.inv(X_T_X)
        self.closed_form_weights = np.dot(np.dot(X_T_X_inv, X_T), y)[1:]
        self.closed_form_intercept = np.dot(np.dot(X_T_X_inv, X_T), y)[0]

    # This function computes the gradient descent solution of linear regression.
    def gradient_descent_fit(self, X, y, lr, epochs):
        # Compute the solution by gradient descent.
        # Save the weights and intercept to self.gradient_descent_weights and self.gradient_descent_intercept
        # lr is the learning rate
        
        num_samples, num_features = X.shape
        self.gradient_descent_weights = np.zeros(num_features)
        self.gradient_descent_intercept = 0
        
        for _ in range(epochs):
            y_pred = self.gradient_descent_predict(X)
            #dw = (-2/num_samples)*np.dot(np.transpose(X),np.subtract(y, y_pred))
            #db = (-2/num_samples)*np.sum(y-y_pred)
            gradient_weights = -(2/num_samples) * np.dot(np.transpose(X), (y - y_pred))
            gradient_intercept = -(2/num_samples) * np.sum(y - y_pred)
            self.gradient_descent_weights -= lr * gradient_weights
            self.gradient_descent_intercept -= lr * gradient_intercept
        #self.gradient_descent_weights=weights
        #self.gradient_descent_intercept=intercept
        

    # This function compute the MSE loss value between your prediction and ground truth.
    def get_mse_loss(self, prediction, ground_truth):
        # Return the value.
        return np.mean((prediction - ground_truth) ** 2)

    # This function takes the input data X and predicts the y values according to your closed-form solution.
    def closed_form_predict(self, X):
        # Return the prediction.
        return np.dot(X, self.closed_form_weights) + self.closed_form_intercept

    # This function takes the input data X and predicts the y values according to your gradient descent solution.
    def gradient_descent_predict(self, X):
        # Return the prediction.
        # print(X)
        return np.dot(X, self.gradient_descent_weights) + self.gradient_descent_intercept
    
    # This function takes the input data X and predicts the y values according to your closed-form solution, 
    # and return the MSE loss between the prediction and the input y values.
    def closed_form_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.closed_form_predict(X), y)

    # This function takes the input data X and predicts the y values according to your gradient descent solution, 
    # and return the MSE loss between the prediction and the input y values.
    def gradient_descent_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.gradient_descent_predict(X), y)
        
    # This function use matplotlib to plot and show the learning curve (x-axis: epoch, y-axis: training loss) of your gradient descent solution.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_learning_curve(self, X, y, lr, epochs):
        num_samples, num_features = X.shape
        self.gradient_descent_weights = np.zeros(num_features)
        self.gradient_descent_intercept = 0
        losses = []
        for _ in range(epochs):
            y_pred = self.gradient_descent_predict(X)
            loss = self.get_mse_loss(y_pred, y)
            losses.append(loss)
            gradient_weights = -(2/num_samples) * np.dot(np.transpose(X), (y - y_pred))
            gradient_intercept = -(2/num_samples) * np.sum(y - y_pred)
            self.gradient_descent_weights -= lr * gradient_weights
            self.gradient_descent_intercept -= lr * gradient_intercept
        #print(losses)
        plt.plot(range(epochs), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Learning Curve')
        plt.show()

# Do not modify the main function architecture.
# You can only modify the arguments of your gradient descent fitting function.
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    
    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    LR.gradient_descent_fit(train_x, train_y, lr=0.00019, epochs=400000)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")
    
    
    


