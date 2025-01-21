# Task Description:
## Task General Description: 

This task will help you practice building and evaluating regression models using scikit-learn. You will implement linear and polynomial regression models and assess them using key evaluation metrics.

### Requirement 1: Linear Regression Implementation

#### Dataset Loading:

Load the California Housing dataset using pandas.

#### Model Implementation:

Use LinearRegression from scikit-learn to create and train a linear regression model.
Split the dataset into training and testing sets (e.g., 80% training, 20% testing).
Train the model by fitting it to the training data.
Use the trained model to make predictions on the test set.
#### Evaluation:

Compute the Mean Squared Error (MSE) of the model's predictions to assess its performance.
Interpret the MSE value to understand the model's fit to the data.
#### Required Deliverable:

Python code implementing the linear regression model.
Output displaying the calculated MSE for the linear regression model.


### Requirement 2: Polynomial Regression Implementation

#### Data Transformation:

Use PolynomialFeatures from scikit-learn to expand the dataset's feature set by creating polynomial features of a specified degree.

#### Model Implementation:

Implement a polynomial regression model using the transformed features.
Train the model on the modified dataset.
Use the trained model to make predictions on the test set.

#### Evaluation:

Compute the MSE for the polynomial regression model.
Compare the MSE values of the linear regression and polynomial regression models to analyze the difference in their performance.


#### Required Deliverable:

Python code implementing the polynomial regression model.
A summary or visualization comparing the MSE values for both models.


### Requirement 3: Visualization of Results

#### Data Visualization:

Use matplotlib to create clear and informative plots:
Plot the original dataset values.
Plot the predictions from the linear regression model.
Plot the predictions from the polynomial regression model.

#### Highlighting Model Differences:

Compare the fit of the linear and polynomial regression models.
Clearly show the discrepancies in how the two models capture the data patterns, especially for nonlinear relationships.

#### Required Deliverable:

A plot comparing the original data and predictions from both models.
The visualization should effectively highlight the differences in the quality of fit between the linear and polynomial regression models.
