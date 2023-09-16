# Titanic Dataset Analysis

This project provides an analysis of the Titanic dataset, exploring factors that may have influenced the survival of passengers on the Titanic. The dataset contains information about passengers, including their age, gender, passenger class, fare, and whether they survived or not.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Analysis and Insights](#analysis-and-insights)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

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

1. Clone this GitHub repository:

   ```bash
   git clone https://github.com/your-username/titanic-dataset-analysis.git
   cd titanic-dataset-analysis
   pip install -r requirements.txt
2. Make sure the .csv is in the directory if not its easy found on kaggle
3. Run the Jupyter Notebook or Python scripts to execute the analysis.
4. Explore the generated visualizations and insights.

## Requirements 
  1. Python 3.x
  2. Jupyter Notebook, optional 
  3. Libraries listed in requirements.txt

## License
This project is licensed under the MIT License. You are free to use, modify, or distribute this code for any purpose.
