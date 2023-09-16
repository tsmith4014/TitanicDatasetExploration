# Titanic Dataset Analysis

This project provides an analysis of the Titanic dataset, exploring factors that may have influenced the survival of passengers on the Titanic. The dataset contains information about passengers, including their age, gender, passenger class, fare, and whether they survived or not.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Analysis and Insights](#analysis-and-insights)
- [Usage](#usage)

## Introduction

The sinking of the Titanic is one of the most famous shipwrecks in history. This project aims to analyze the Titanic dataset to uncover insights into the factors that contributed to passenger survival. It includes statistical tests, visualizations, and machine learning models to explore relationships between various attributes and survival rates.

## Dataset Overview

The dataset used in this project contains the following columns:

- `PassengerId`: A unique identifier for each passenger.
- `Survived`: 1 if the passenger survived, 0 otherwise.
- `Pclass`: The passenger class (1st, 2nd, or 3rd).
- `Sex`: The gender of the passenger.
- `Age`: The age of the passenger.
- `SibSp`: The number of siblings or spouses aboard.
- `Parch`: The number of parents or children aboard.
- `Fare`: The fare paid by the passenger.
- `Embarked`: The port of embarkation.

## Titles in the Titanic Dataset

In the Titanic dataset, passenger titles are extracted from their names and are used as a form of addressing or categorizing passengers. These titles provide additional information about the passengers, such as their social status, gender, or occupation, which can be useful for analyzing survival rates or making inferences about the demographics of passengers on the Titanic.

Here's a brief explanation of each title:

- **Mr**: A common English honorific for men.
- **Mrs**: An honorific for married or adult women.
- **Miss**: An honorific for unmarried women or girls.
- **Master**: A title used to address boys, especially those who are young.
- **Don**: A Spanish and Portuguese honorific, typically used for a nobleman or gentleman.
- **Rev**: An abbreviation for "Reverend," used for ministers, priests, or clergy members.
- **Dr**: An abbreviation for "Doctor," used for medical doctors or individuals with doctoral degrees.
- **Mme**: An abbreviation for "Madame," used for married or adult women, similar to "Mrs."
- **Ms**: A title that doesn't specify marital status, often used for women.
- **Major**: A military rank.
- **Lady**: A title of nobility, often used for women of high social standing.
- **Sir**: A title of honor for men, typically knights or individuals with knighthood.
- **Mlle**: An abbreviation for "Mademoiselle," used for unmarried women or girls, similar to "Miss."
- **Col**: An abbreviation for "Colonel," a military rank.
- **Capt**: An abbreviation for "Captain," often used for the leader or commander of a ship.
- **Countess**: A noble title, often the wife of an Earl or Count.
- **Jonkheer**: A Dutch honorific for young noblemen or esquires.

These titles add depth to the analysis of the Titanic dataset and can help researchers understand the demographics and social structure of the passengers aboard the Titanic.

## Analysis and Insights

1. Descriptive Statistics: Basic statistics about passenger age and fare.

2. Chi-Squared Tests: Assess the relationship between survival and passenger gender and class.

3. Logistic Regression: Build a logistic regression model to predict survival based on age, gender, and class.

4. Random Forest Classifier: Train a random forest classifier to predict survival and evaluate its accuracy.

5. Visualizations: Histograms and bar plots to visualize age and family size's impact on survival.

6. Correlation Analysis: Calculate point-biserial correlations between age/fare and survival.

7. T-Tests: Determine if differences in mean fares are significant between various groups.

## Usage

To run the analysis code and explore the project, follow these steps:

1. Set up your virtual env
   ```bash
   python -m venv venv
   source venv/bin/activate
   git clone https://github.com/your-username/titanic-dataset-analysis.git
   cd titanic
   pip install -r requirements.txt

2. Using VScode/Python to execute the analysis.
3. Explore the generated visualizations and insights.

## Predictive Models

In this project, we've created two predictive models to estimate the likelihood of survival for Titanic passengers based on various features:

### Logistic Regression

We fitted a logistic regression model to predict survival. This model takes into account passenger age, gender, class, and fare. Here's a summary of the logistic regression model::

```python
X = titanic_df[['Sex', 'Pclass', 'Age', 'Fare']]
X = sm.add_constant(X)  # Add a constant for the intercept
y = titanic_df['Survived']

# Fit logistic regression model
logistic_model = sm.Logit(y, X)
logistic_results = logistic_model.fit()

# Get the summary of the logistic regression to view coefficients
coefficients_summary = logistic_results.summary()

print(coefficients_summary)

```

The output provides insights into the impact of each variable on survival.

Logistic Regression with 'Age' and 'Fare' as Predictors
We also explored a logistic regression model using only 'Age' and 'Fare' as predictors to investigate their influence on survival:

```python
# Add a constant to the independent variables (intercept)
X_with_const = sm.add_constant(titanic_df[['Age', 'Fare']])
logit_model = sm.Logit(titanic_df['Survived'], X_with_const)
logit_results = logit_model.fit()

# Get the summary of the logistic regression
print(logit_results.summary())

```

This model focuses on age and fare's effect on survival.

Logistic Regression Model with Titles
We examined how passenger titles influenced survival rates. Here's how to perform this analysis:

```python
# Extract titles from passenger names
titanic_df['Title'] = titanic_df['Name'].str.extract(' ([A-Za-z]+)\.')

# Analyze survival rates by title
title_survival_rates = titanic_df.groupby('Title')['Survived'].mean().sort_values(ascending=False)

# Visualize survival rates by title
```
