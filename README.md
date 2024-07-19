

# Regression Techniques in Machine Learning


## Linear Regression for Bandgap Prediction

This repository contains a project focused on predicting the bandgap of molecules using linear regression. The project utilizes a dataset with four independent variables representing molecular properties to predict the bandgap (dependent variable).

## Objective

The objective is to apply linear regression techniques to understand the relationship between molecular properties and their bandgap values. This is a common problem in materials science, where understanding the electronic properties of materials is crucial for various applications.

## Dataset Description

The dataset used in this project includes the following features for each molecule:
- Four independent variables (floating-point numbers) representing different molecular properties.
- One dependent variable (floating-point number) representing the bandgap of the molecule.



## Theoretical Background

### Linear Regression

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. The model assumes a linear relationship between the variables, expressed as:

<p> y = &beta;<sub>0</sub> + &beta;<sub>1</sub>x<sub>1</sub> + &beta;<sub>2</sub>x<sub>2</sub> + &hellip; + &beta;<sub>n</sub>x<sub>n</sub></p>

Where:
- <span> y </span> is the dependent variable (bandgap).
- <span> x<sub>1</sub>, x<sub>2</sub>, &hellip;, x<sub>n</sub> </span> are the independent variables (molecular properties).
- <span> &beta;<sub>0</sub> </span> is the intercept.
- <span> &beta;<sub>1</sub>, &beta;<sub>2</sub>, &hellip;, &beta;<sub>n</sub> </span> are the coefficients of the independent variables.

### Loss Function

The loss function used in linear regression is the Mean Squared Error (MSE), which measures the average squared difference between the actual and predicted values:


<p> MSE = (1/n) &sum; (y<sub>i</sub> - &ycirc;<sub>i</sub>)<sup>2</sup> </p>


Where:
- <span> y<sub>i</sub> </span> is the actual value of the dependent variable for the <span> i </span>-th observation.
- <span> &ycirc;<sub>i</sub> </span> is the predicted value of the dependent variable for the <span> i </span>-th observation.
- <span> n </span> is the number of observations.

### Gradient Descent

To minimize the MSE and find the optimal coefficients (<span> &beta; </span>), we use gradient descent, an iterative optimization algorithm. The idea is to update the coefficients in the opposite direction of the gradient of the loss function with respect to the coefficients. The update rule for each coefficient <span> &beta;<sub>j</sub> </span> is:

![image1](images/image1.png)



Where:
- <span> &alpha; </span> is the learning rate, a small positive number that determines the step size.

### Derivative Calculation

The gradient of the MSE with respect to each coefficient <span> &beta;<sub>j</sub> </span> is calculated as follows:


![image1](images/image2.png)



This simplifies to:


![image1](images/image3.png)


The intercept term <span> &beta;<sub>0</sub> </span> is updated as:

![image1](images/image4.png)


Which simplifies to:

![image1](images/image5.png)


### Methodology

1. **Data Preprocessing:**
    - **Loading the Dataset:** The dataset is loaded from a text file.
    - **Splitting the Dataset:** The dataset is split into training and testing sets to evaluate the model's performance on unseen data.

2. **Linear Regression Model:**
    - **Implementing from Scratch:** A linear regression model is implemented from scratch without using inbuilt functions, providing a deeper understanding of the underlying algorithm.
    - **Training the Model:** The model is trained using the training dataset, where the optimal coefficients (<span> &beta; </span>) are estimated by minimizing the Mean Squared Error (MSE) using gradient descent.
    - **Making Predictions:** The model makes predictions on the testing dataset based on the learned coefficients.

3. **Evaluation:**
    - **Performance Metrics:** The model's performance is evaluated using metrics such as Mean Squared Error (MSE) and R-squared (<span> R^2 </span>):<p></p>
     ![image1](images/image6.png)

    - **Visualization:** The predicted bandgap values are compared with the actual values through visualizations.

The parity plot obtained is as follows:

![image1](images/graph1.png)

# Logistic Regression 

This repository also implements logistic regression from scratch. The objective is to gain a deep understanding of the logistic regression algorithm by using numpy and other math libraries.

## Introduction
Logistic regression is a fundamental machine learning algorithm used for binary classification tasks. It models the probability that a given input belongs to a particular class. The logistic function (also known as the sigmoid function) is used to map predicted values to probabilities.

## Theory
### Logistic Function
The logistic function is defined as:

<p>&sigma;(z) = <sup>1</sup>&frasl;<sub>1 + e<sup>-z</sup></sub></p>

where <sigma> z </sigma> is the input to the function. The output of the logistic function is always between 0 and 1, making it suitable for binary classification.

### Cost Function
The cost function for logistic regression is the log-loss (cross-entropy loss):

<p>J(&theta;) = -<sup>1</sup>&frasl;<sub>m</sub> &sum;<sub>i=1</sub><sup>m</sup> [ y<sub>(i)</sub> log(h<sub>&theta;</sub>(x<sub>(i)</sub>)) + (1 - y<sub>(i)</sub>) log(1 - h<sub>&theta;</sub>(x<sub>(i)</sub>)) ]</p>


where <span> m </span> is the number of training examples, <span>y<sub>(i)</sub></span> is the true label, and <span>h<sub>&theta;</sub>(x<sub>(i)</sub>)</span> is the predicted probability.

### Gradient Descent
To minimize the cost function, we use gradient descent. The update rule for gradient descent is:
<p>&theta; := &theta; - &alpha; <sup>&part; J(&theta;)</sup>&frasl;<sub>&part; &theta;</sub></p>

where <span>&alpha;</span> is the learning rate.

## Implementation
The implementation involves the following steps:
1. **Data Generation**: Using `make_blobs` from sklearn to create a synthetic dataset.
2. **Sigmoid Function**: Implementing the logistic (sigmoid) function.
3. **Cost Function**: Calculating the cost using the log-loss function.
4. **Gradient Descent**: Implementing gradient descent to optimize the parameters.
5. **Prediction**: Using the trained model to make predictions on new data.
6. **Evaluation**: Assessing the performance of the model using metrics like accuracy.

Below is the decision boundary obtained after training the logistic regression model:

![image1](images/graph2.png)