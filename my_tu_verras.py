import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn import metrics


def load_dataset():
    file = ("boston.csv")
    dataset = pd.read_csv(file)
    return dataset


load_dataset()

def print_summarize_dataset(dataset):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
    print("Dataset dimension:")
    print(dataset.shape)
    print()
    print("First 10 rows of dataset:")
    print(dataset.head(10))
    print()
    print("Statistical summary:")
    print(dataset.describe())
    
dataset = load_dataset()          
print_summarize_dataset(dataset)

def clean_dataset(dataset):
    # Print if there are null values in the dataset
    print(dataset.isna().any())
    # remove all rows that contain null values
    dataset_cleaned = dataset.dropna()
    return dataset_cleaned
    

dataset_cleaned = clean_dataset(dataset)


def print_histograms(dataset_cleaned):
    fig, axs = plt.subplots(4, 4, figsize=(14, 10))  
    axs = axs.ravel()  

    # Plot histograms on each subfigure
    for i, column in enumerate(dataset_cleaned.columns):
        ax = axs[i]
        ax.hist(dataset_cleaned[column], bins=10)  
        ax.set_title('{}'.format(column))
        # Turn on the grid for the subplot
        ax.grid(True)
        
    # Adjust the spacing between subplots
    fig.tight_layout()
    plt.show()


print_histograms(dataset_cleaned)


def compute_correlations_matrix(dataset_cleaned):
    correlations = dataset_cleaned.corr(method='pearson')
    return correlations


correlations = compute_correlations_matrix(dataset_cleaned)
print(correlations["MDEV"])

def print_scatter_matrix(dataset_cleaned):
    # Plot every attribute against each other
    pd.plotting.scatter_matrix(dataset_cleaned, figsize=(30,30))
    plt.show()
    # Plot MEDV in function of RM
    plt.scatter(dataset_cleaned["MDEV"], dataset_cleaned["RM"])
    plt.xlabel('RM')
    plt.ylabel('MDEV')
    plt.show()
    # Plot the correlation scatter plot of the median value against LSTAT, AGE, and CRIME
    plt.scatter(dataset_cleaned["LSTAT"], dataset_cleaned["MDEV"])
    plt.xlabel("LSTAT")
    plt.ylabel("MDEV")
    plt.show()
    
    plt.scatter(dataset_cleaned["AGE"], dataset_cleaned["MDEV"])
    plt.xlabel("AGE")
    plt.ylabel("MDEV")
    plt.show()
    
    plt.scatter(dataset_cleaned["CRIM"], dataset_cleaned["MDEV"])
    plt.xlabel("CRIM")
    plt.ylabel("MDEV")
    plt.show()
    
    # Plot the scatter matrix or print the correlation coefficients for LSTAT
    lstat_corr = dataset_cleaned.corr()["LSTAT"]
    pd.plotting.scatter_matrix(dataset_cleaned[["LSTAT", "RM", "AGE", "CRIM"]], figsize=(10,10))
    plt.show()
    

print_scatter_matrix(dataset_cleaned)      


def boston_fit_model(dataset_cleaned):
    model_dataset = dataset_cleaned[["RM", "MDEV"]]
    regressor = sklearn.linear_model.LinearRegression()
    # Exctract column 1
    x = model_dataset.iloc[:, :-1].values
    # Exctract column 2
    y = model_dataset.iloc[:, 1].values
    # Train the model
    regressor.fit(x, y)
    return regressor


def boston_predict(estimator, array_to_predict):
    # Make sure the input is a 2D array
    X = np.array(array_to_predict).reshape(1, -1)
    y_pred = estimator.predict(X)
    return y_pred

estimator = boston_fit_model(dataset_cleaned)
data = [5]
print(boston_predict(estimator, data))
