# Task Description:
## Task Title: Binary Classification Model for Breast Cancer Detection using PyTorch

### Task Description:
Build a binary classification model using PyTorch to accurately classify breast cancer tumors as malignant (1) or benign (0) based on various features from the Breast Cancer dataset provided by sklearn.

### Requirement 1 Title: Setup and Data Preparation

#### Requirement 1 Description:
- Load the Breast Cancer dataset from sklearn.datasets.
- Use all numerical features for training and convert the target variable into binary values where malignant tumors (M) are marked as 1 and benign tumors (B) are marked as 0.
- Split the dataset into a 70% training set and a 30% test set, and normalize the feature values using StandardScaler.

#### Requirement 1 Deliverables:

- Python script or notebook with code for loading the Breast Cancer dataset.
- Code for splitting the dataset into training and test sets.
- Code for normalizing the feature values.

### Requirement 2 Title: Model Building

#### Requirement 2 Description:
Create a fully connected neural network that includes:
- An input layer that matches the number of features.
- Two hidden layers with ReLU activation to facilitate non-linear learning.
- An output layer with Sigmoid activation for binary classification, suitable for outputting probabilities for two classes.

#### Requirement 2 Deliverables:

- Python script or notebook with code for constructing the neural network architecture.

### Requirement 3 Title: Training the Model

#### Requirement 3 Description:
- Configure the model with Binary Cross-Entropy Loss (BCELoss) for binary classification and Adam optimizer for efficient training. - Set the model to train for 100 epochs.
- track the loss during training, and plot training and validation loss curves to visualize the learning process.

#### Requirement 3 Deliverables:

- Python script or notebook with code for compiling and training the model.
- Visualizations of loss curves showing training and validation loss over epochs.

### Requirement 4 Title: Model Evaluation and Prediction

#### Requirement 4 Description:
- Evaluate the trained model on the test set to measure its performance using metrics such as test accuracy and F1-score.
- Generate and display a confusion matrix both numerically and visually, and create a scatter plot of actual vs. predicted classifications to assess model predictions visually.

#### Requirement 4 Deliverables:

- Python script or notebook with code for evaluating the model and making predictions.
- Visualizations including a plotted confusion matrix.
- Scatter plot showing actual vs. predicted classifications.

### Dataset:
- Breast Cancer dataset from sklearn.