# Credit_Card_Approval_Explainability

## Overview
This project focuses on predicting credit card approval using a machine learning model (Multilayer Perceptron, MLP) and interpreting the model's predictions using explainability techniques such as **LIME** (Local Interpretable Model-agnostic Explanations) and **SHAP** (SHapley Additive exPlanations).

The dataset used contains customer features such as income, family size, and mortgage details, with the goal of predicting whether a customer will be approved for a credit card. This project also compares the interpretability of LIME and SHAP in explaining model outcomes.

## Features
The dataset contains the following features:
- **ID**: Customer ID
- **Age**: Customer Age
- **Experience**: Work experience (in years)
- **Income**: Annual income (in thousands)
- **Zipcode**: Residential area zipcode
- **Family**: Number of family members
- **CCAvg**: Average monthly credit card spending
- **Education**: Education level (1: Bachelor, 2: Master, 3: Advanced Degree)
- **Mortgage**: Mortgage (in thousands)
- **Securities Account**: Boolean indicating if the customer has a securities account
- **CD Account**: Boolean indicating if the customer has a Certificate of Deposit account
- **Online**: Boolean indicating if the customer uses online banking
- **CreditCard**: Target column, indicates whether a credit card is approved (1: Yes, 0: No)

## Project Tasks

### Task 1: Data Loading and Exploratory Data Analysis (EDA)
- Load the dataset and perform basic exploratory data analysis (EDA).
- Visualize and normalize features where appropriate.

### Task 2: MLP Model Implementation
- Build an MLP model with no more than 2 hidden layers.
- Perform 5-fold cross-validation to evaluate model performance.
- Report training error and cross-validation error.

### Task 3: LIME Explainability
- Select 5 random data points and apply **LIME** to explain the individual predictions.
- Implement submodular pick and generate LIME explanations for 10% of the training data with up to 5 explanations.
- Use these explanations to predict credit card approval on the entire dataset and calculate the classification error.

### Task 4: SHAP Explainability
- For the same 5 data points selected in Task 3, apply **SHAP** to explain the predictions.

### Task 5: Comparison and Observations
- Compare the insights obtained from **LIME** and **SHAP**.
- Share key observations on the explainability of the MLP predictions.

## Installation
To run this project, you need the following dependencies:
- Python 3.x
- pandas
- numpy
- scikit-learn
- tensorflow
- lime
- shap
- matplotlib
- seaborn

Install the dependencies via pip:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Credit_Card_Approval_Explainability.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Credit_Card_Approval_Explainability
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook FATML_assignment2_Group_26.ipynb
   ```

## Results
The results and visualizations of LIME and SHAP will be displayed within the notebook. Additionally, observations on the differences between LIME and SHAP will be discussed in the notebook.
