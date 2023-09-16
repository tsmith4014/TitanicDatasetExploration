import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, pointbiserialr, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the Titanic dataset
titanic_df = pd.read_csv('titanic.csv')

# Exploratory Data Analysis (EDA)

# Print the column names
column_names = titanic_df.columns
print("Column Names:")
print(column_names)

# Count the number of male and female passengers
gender_counts = titanic_df["Sex"].value_counts()
males = gender_counts['male']
females = gender_counts['female']
total_passengers = males + females
print(f"Total number of male and female passengers: {total_passengers}")

# Calculate the average fare paid by passengers
avg_fare = titanic_df["Fare"].mean()
print(f"Average fare paid by passengers: {avg_fare:.2f}")

# Calculate the average age of passengers
avg_age = titanic_df["Age"].mean()
print(f"Average age of passengers: {avg_age:.2f}")

# Count the total number of first-class passengers (Pclass = 1)
fc_passengers = titanic_df[titanic_df["Pclass"] == 1]
total_fc_passengers = len(fc_passengers)
print(f"Total number of passengers in first class (Pclass = 1): {total_fc_passengers}")

# Count the total number of survivors in first class
fc_survivors = fc_passengers[fc_passengers["Survived"] == 1]
total_fc_survivors = len(fc_survivors)
print(f"Total number of survivors in first class (Pclass = 1): {total_fc_survivors}")

# Count the total number of female and male survivors in first class
fc_female_survivors = fc_survivors[fc_survivors["Sex"] == "female"]
fc_male_survivors = fc_survivors[fc_survivors["Sex"] == "male"]
total_fc_female_survivors = len(fc_female_survivors)
total_fc_male_survivors = len(fc_male_survivors)

print(f"Total number of female survivors in first class: {total_fc_female_survivors}")
print(f"Total number of male survivors in first class: {total_fc_male_survivors}")

# Calculate survivors by passenger class
survivors_by_class = titanic_df.groupby("Pclass")["Survived"].sum()
print("\nSurvivors by Passenger Class:")
print(survivors_by_class)

# Descriptive statistics for 'Age' and 'Fare'
age_stats = titanic_df['Age'].describe()
fare_stats = titanic_df['Fare'].describe()
print("\nDescriptive Statistics for 'Age':")
print(age_stats)
print("\nDescriptive Statistics for 'Fare':")
print(fare_stats)

# Chi-squared test for 'Sex' and 'Pclass' against 'Survived'
def perform_chi_squared_test(variable):
    observed = pd.crosstab(titanic_df['Survived'], titanic_df[variable])
    chi2, p, _, _ = chi2_contingency(observed)
    alpha = 0.05

    print(f"\nChi-squared test for {variable}:")
    print(f"Chi-squared statistic: {chi2:.2f}")
    print(f"P-value: {p:.4f}")

    if p < alpha:
        print(f"There is evidence to suggest a significant relationship between {variable} and survival.")
    else:
        print(f"There is no significant evidence to suggest a relationship between {variable} and survival.")

# Perform chi-squared tests for 'Sex' and 'Pclass'
perform_chi_squared_test("Sex")
perform_chi_squared_test("Pclass")

# Impute missing 'Age' values with the median age
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Prepare data for logistic regression (encode categorical variables)
X = pd.get_dummies(titanic_df[['Sex', 'Pclass', 'Age']], drop_first=True)
y = titanic_df['Survived']

# Fit logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a random forest classifier
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = random_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Classifier:")
print(f"Accuracy: {accuracy:.2%}")

# Visualize data

# Histogram of 'Age' by 'Survived'
sns.histplot(data=titanic_df, x='Age', hue='Survived', bins=20, kde=True)
plt.title('Distribution of Age by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Feature Engineering

# Create a new feature 'FamilySize' by adding 'SibSp' and 'Parch'
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch']

# Visualize the impact of family size on survival (e.g., bar plot)
sns.barplot(data=titanic_df, x='FamilySize', y='Survived')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Family Size')
plt.show()

# Descriptive statistics for 'Fare'
fare_stats = titanic_df.groupby('Survived')['Fare'].describe()
print("\nDescriptive Statistics for 'Fare':")
print(fare_stats)

# Histogram of 'Fare' by 'Survived'
plt.figure(figsize=(8, 6))
sns.histplot(data=titanic_df, x='Fare', hue='Survived', bins=30, kde=True)
plt.title('Distribution of Fare by Survival')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# Calculate point-biserial correlation between 'Age' and 'Survived'
corr_age_survived, p_age = pointbiserialr(titanic_df['Age'], titanic_df['Survived'])
print(f"\nPoint-biserial correlation between 'Age' and 'Survived': {corr_age_survived:.2f}, p-value: {p_age:.4f}")

# Calculate point-biserial correlation between 'Fare' and 'Survived'
corr_fare_survived, p_fare = pointbiserialr(titanic_df['Fare'], titanic_df['Survived'])
print(f"Point-biserial correlation between 'Fare' and 'Survived': {corr_fare_survived:.2f}, p-value: {p_fare:.4f}")

# Additional Analysis: Explore the hypothesis that younger passengers survived because they paid higher fares.

# Define the age threshold for younger passengers
age_threshold = 18

# Select younger survivors and younger non-survivors
young_survivors = titanic_df[(titanic_df['Age'] < age_threshold) & (titanic_df['Survived'] == 1)]
young_non_survivors = titanic_df[(titanic_df['Age'] < age_threshold) & (titanic_df['Survived'] == 0)]

# Calculate the mean fare for younger survivors and younger non-survivors
mean_fare_young_survivors = young_survivors['Fare'].mean()
mean_fare_young_non_survivors = young_non_survivors['Fare'].mean()

print(f"\nMean Fare for Young Survivors (Age < {age_threshold}): {mean_fare_young_survivors:.2f}")
print(f"Mean Fare for Young Non-Survivors (Age < {age_threshold}): {mean_fare_young_non_survivors:.2f}")

# Perform a t-test to determine if the difference in mean fares is significant
t_stat, p_value_fare_age = ttest_ind(young_survivors['Fare'], young_non_survivors['Fare'], equal_var=False)
print(f"T-test for Fare Difference among Young Passengers (Age < {age_threshold}):")
print(f"T-statistic: {t_stat:.2f}, p-value: {p_value_fare_age:.4f}")

if p_value_fare_age < 0.05:
    print("There is evidence to suggest a significant difference in mean fares between young survivors and young non-survivors.")
else:
    print("There is no significant evidence to suggest a difference in mean fares between young survivors and young non-survivors.")

# Explore the impact of fare on survival for older passengers

# Define the age threshold for older passengers
age_threshold_old = 50

# Select older survivors and older non-survivors
older_survivors = titanic_df[(titanic_df['Age'] >= age_threshold_old) & (titanic_df['Survived'] == 1)]
older_non_survivors = titanic_df[(titanic_df['Age'] >= age_threshold_old) & (titanic_df['Survived'] == 0)]

# Calculate the mean fare for older survivors and older non-survivors
mean_fare_older_survivors = older_survivors['Fare'].mean()
mean_fare_older_non_survivors = older_non_survivors['Fare'].mean()

print(f"Mean Fare for Older Survivors (Age >= {age_threshold_old}): {mean_fare_older_survivors:.2f}")
print(f"Mean Fare for Older Non-Survivors (Age >= {age_threshold_old}): {mean_fare_older_non_survivors:.2f}")

# Perform a t-test to determine if the difference in mean fares is significant
t_stat_older, p_value_fare_age_older = ttest_ind(older_survivors['Fare'], older_non_survivors['Fare'], equal_var=False)
print(f"T-test for Fare Difference among Older Passengers (Age >= {age_threshold_old}):")
print(f"T-statistic: {t_stat_older:.2f}, p-value: {p_value_fare_age_older:.4f}")

if p_value_fare_age_older < 0.05:
    print("There is evidence to suggest a significant difference in mean fares between older survivors and older non-survivors.")
else:
    print("There is no significant evidence to suggest a difference in mean fares between older survivors and older non-survivors.")

# Logistic Regression Model with 'Age' and 'Fare' as Predictors

# Add a constant to the independent variables (intercept)
X_with_const = sm.add_constant(titanic_df[['Age', 'Fare']])
logit_model = sm.Logit(titanic_df['Survived'], X_with_const)
logit_results = logit_model.fit()

# Get the summary of the logistic regression
print(logit_results.summary())

# Explore the impact of passenger titles on survival

# Extract titles from passenger names
titanic_df['Title'] = titanic_df['Name'].str.extract(' ([A-Za-z]+)\.')

# Explore unique titles
unique_titles = titanic_df['Title'].unique()
print("Unique Titles:")
print(unique_titles)

# Analyze survival rates by title
title_survival_rates = titanic_df.groupby('Title')['Survived'].mean().sort_values(ascending=False)
print("\nSurvival Rates by Title:")
print(title_survival_rates)

# Visualize survival rates by title
plt.figure(figsize=(10, 6))
sns.barplot(data=titanic_df, x='Title', y='Survived', order=title_survival_rates.index)
plt.xlabel('Title')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Title')
plt.xticks(rotation=45)
plt.show()

# Create a contingency table of titles and survival outcomes
contingency_table = pd.crosstab(titanic_df['Title'], titanic_df['Survived'])

# Perform the chi-squared test
chi2, p, _, _ = chi2_contingency(contingency_table)

# Define the significance level
alpha = 0.05

print(f"Chi-squared statistic: {chi2:.2f}")
print(f"P-value: {p:.4f}")

if p < alpha:
    print("There is evidence to suggest a significant relationship between titles and survival.")
else:
    print("There is no significant evidence to suggest a relationship between titles and survival.")

# Check unique values in the 'Title' column
unique_titles = titanic_df['Title'].unique()
print("Unique Titles:")
print(unique_titles)

# Check for missing values in the 'Title' column
missing_titles = titanic_df['Title'].isnull().sum()
print(f"Missing Titles: {missing_titles}")

# Check data types of all columns
# column_data_types = titanic_df.dtypes
# print(column_data_types)

# Model Building and Evaluation

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Random Forest Model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions using both models
logistic_predictions = logistic_model.predict(X_test)
random_forest_predictions = random_forest_model.predict(X_test)

# Evaluate model accuracy
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)

print("\nModel Accuracies:")
print(f"Logistic Regression Accuracy: {logistic_accuracy:.2%}")
print(f"Random Forest Accuracy: {random_forest_accuracy:.2%}")

# You can now use these models for predictions on new data.


# Encode 'Sex' as 0 for male and 1 for female
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})

# Prepare data for logistic regression (include 'Sex', 'Pclass', 'Age', and 'Fare')
X = titanic_df[['Sex', 'Pclass', 'Age', 'Fare']]
X = sm.add_constant(X)  # Add a constant for the intercept
y = titanic_df['Survived']

# Fit logistic regression model
logistic_model = sm.Logit(y, X)
logistic_results = logistic_model.fit()

# Get the summary of the logistic regression to view coefficients
coefficients_summary = logistic_results.summary()

print(coefficients_summary)


# Extract coefficients and variable names
coefficients = logistic_results.params
variables = coefficients.index

# Plot the coefficients
plt.figure(figsize=(10, 6))
plt.barh(variables, coefficients)
plt.xlabel('Coefficient Value')
plt.ylabel('Variable')
plt.title('Logistic Regression Coefficients')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()



#final model outpiut explained, refer to the output you received from the logistic regression model summary in your terminal.  Insightful stuff!

# Sex: Coefficient = 2.6073

# Interpretation: The coefficient for 'Sex' is 2.6073, which is highly positive. This means that being female (coded as 1) significantly increases the log-odds of survival compared to being male (coded as 0). To interpret it further, you can exponentiate this coefficient to get the odds ratio. Exponentiating 2.6073 gives you approximately 13.57, which means that females have about 13.57 times higher odds of survival compared to males when other variables are held constant.
# Pclass: Coefficient = -1.1529

# Interpretation: The coefficient for 'Pclass' is negative, indicating that as the passenger class level increases (from 1st class to 2nd class to 3rd class), the log-odds of survival decrease. Exponentiating this coefficient shows that for each increase in class level, the odds of survival decrease by approximately 0.316 times (or about 31.6%).
# Age: Coefficient = -0.0331

# Interpretation: The coefficient for 'Age' is negative, indicating that as age increases, the log-odds of survival decrease. However, the magnitude of this effect is relatively small. For each one-unit increase in age, the odds of survival decrease by approximately 3.35%. This suggests that older passengers are slightly less likely to survive.
# Fare: Coefficient = 0.0006

# Interpretation: The coefficient for 'Fare' is positive but very close to zero. This suggests that the fare paid by passengers has a minimal impact on the log-odds of survival. In practical terms, it means that fare is not a strong predictor of survival in this model.
# In summary, based on this logistic regression model:

# Being female (Sex = 1) significantly increases the odds of survival compared to being male (Sex = 0).
# Higher passenger class (Pclass) significantly decreases the odds of survival as you move from higher classes to lower classes.
# Older age (Age) has a small negative impact on the odds of survival.
# Fare does not have a significant impact on the odds of survival in this model.