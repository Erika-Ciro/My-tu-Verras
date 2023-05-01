# My-tu-Verras

# Description
House price is a recurring subject nowadays. In this project, you will build a model to predict the price base on predefined criteria.

We are providing a dataset, the Boston housing prices.


This project is divided into two parts:
1. Understand the data.
2. Build a linear regression prediction model.

# First Glance

Let's get down to the brass task.

From a quick look, we know what is inside our dataset, like the different attributes:

• **CRIM** per capita crime rate by town.

• **ZN** proportion of residential land zoned for lots over 25,000 sq.ft.

• **INDUS** proportion of non-retail business acres per town.

• **CHAS** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

• **NOX** nitric oxides concentration (parts per 10 million).

• **RM** average number of rooms per dwelling.

• **AGE** proportion of owner-occupied units built before 1940.

• **DIS** weighted distances to five Boston employment centers.

• **RAD** index of accessibility to radial highways.

• **TAX** full-value property-tax rate per $10,000.

• **PTRATIO** pupil-teacher ratio by town.

• **B** 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

• **LSTAT** PERCENT lower status of the population.

• **MEDV** Median value of owner-occupied homes is $ 1000's.

Now that we loaded the data let's peek at it.

-> ***Load the data in a data frame and print the first rows***.

# Cleaning and Pre-processing

-> ***Make a function that cleans the dataset from lines which any missing values***.

After this function, we should have **zero** missing values in the dataset.


# Data Analysis

Now we are ready to cut to the chase. We will see if our supposition was correct by visualizing the data.

Visualization is critical to understanding the relationship between the target and the other features.

A great way to get a feel of the data we are dealing with is to plot each attribute in a histogram.

-> ***Plot each attribute in a histogram***.


**Looking for correlations**

So far, we have only taken a quick peek to get a general understanding of the kind of data we are manipulating. Now the goal will be to investigate deeper again.

The size of our dataset is not too large, so we can try to analyze linear correlations between every pair of attributes.

-> ***Write a function compute_correlations_matrix to compute Pearson's correlations between every pair of attributes***.

When the coefficient is close to 1 (in absolute value), there is a strong correlation between the two variables. If it is positive, it means the linear correction is positive. If it is negative... you get it.
Coefficients close to zero mean there is no linear correlation between the attributes.

-> ***What is the correlation coefficient between the median value and the number of rooms?***

This coefficient is positive and is the biggest. It means that the median value increases when the number of rooms increases. Well, that makes sense; we expect a house with many rooms to be more expensive than a single-room apartment.

-> ***Analyse the correlations between the median value and the other attributes. Which attribute is the most negatively correlated with the median value? Does it make sense to you?***

Mathematically, all this information can be seen in an object, the covariance matrix. The covariance matrix tells us a lot about our data's relationship and distribution.

Numbers are cool, but we, as human beings, love colors, pictures, and visual representations. We can easily spot correlations by visualizing the relationship between attributes on graphs.

-> ***Plot every attribute against each other***.
-> ***Make a function that prints the scatter matrix***.
-> ***Plot MEDV in function of RM***.

# Prediction

After exploring the data and gaining more insights, the next step is to do another round of data cleaning and find a model to predict the MEDV.

We are going to use a linear regression algorithm from sklearn.

-> ***Make a function that return a Fitted Estimator***.

-> ***Make a second function that will receive the Fitted Estimator and return a prediction***.
