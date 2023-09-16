# # Total number of male and female passengers.
# # Average fare paid by passengers.
# # Average age of passengers.
# # Total number of survivors in class 1.

import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, pointbiserialr 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# #using pandas

titanic_df = pd.read_csv('titanic.csv')
# print(titanic_df.head())

# Print the column names
column_names = titanic_df.columns
print(column_names)


female_and_male = titanic_df["Sex"].value_counts()

males = female_and_male['male']
female = female_and_male['female']
sum_sexes = males + female
print(f"This is the total amount of male and female passengers:  {sum_sexes}")



#avg fare paid by passengers
avg_fare = titanic_df["Fare"].mean()
print(f"This is the average fare paid by passengers:  {avg_fare}")

#avg age of passengers
avg_age = titanic_df["Age"].mean()
print(f"This is the average age of passengers:  {avg_age}")


#total of first class passengers
fc_passengers = titanic_df[(titanic_df["Pclass"] == 1)]
lenfc_passengers = len(fc_passengers)
print(f"This is the total number of passengers in class 1:  {lenfc_passengers}")

#total number of survivors in class 1
fc_survior = titanic_df[(titanic_df["Survived"] == 1) & (titanic_df["Pclass"] == 1)]
lenoftitanic_df = len(fc_survior)
print(f"This is the total number of survivors in class 1:  {lenoftitanic_df}")

#totol number of survivors in class 1: female, male
fc_female = titanic_df[(titanic_df["Survived"] == 1) & (titanic_df["Pclass"] == 1) & (titanic_df["Sex"] == "female")]
fc_male = titanic_df[(titanic_df["Survived"] == 1) & (titanic_df["Pclass"] == 1) & (titanic_df["Sex"] == "male")]
lenoffc_female = len(fc_female)
lenoffc_male = len(fc_male)
print(f"This is the total number of female survivors in class 1:  {lenoffc_female}")
print(f"This is the total number of male survivors in class 1:  {lenoffc_male}")

# Survivors by Class
survivors_by_class = titanic_df.groupby("Pclass")["Survived"].sum()
print("Survivors by Class:")
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

# Histogram of 'Age' by 'Survived'
sns.histplot(data=titanic_df, x='Age', hue='Survived', bins=20, kde=True)
plt.title('Distribution of Age by Survival')
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




# # Calculate point-biserial correlation between 'Survived' and 'Age'
# from scipy.stats import pointbiserialr

# corr_age_survived, p_age = pointbiserialr(titanic_df['Age'], titanic_df['Survived'])
# print(f"\nPoint-biserial correlation between 'Age' and 'Survived': {corr_age_survived:.2f}, p-value: {p_age:.4f}")

# # Calculate point-biserial correlation between 'Survived' and 'Fare'
# corr_fare_survived, p_fare = pointbiserialr(titanic_df['Fare'], titanic_df['Survived'])
# print(f"Point-biserial correlation between 'Fare' and 'Survived': {corr_fare_survived:.2f}, p-value: {p_fare:.4f}")

# Calculate point-biserial correlation between 'Age' and 'Survived'
corr_age_survived, p_age = pointbiserialr(titanic_df['Age'], titanic_df['Survived'])
print(f"\nPoint-biserial correlation between 'Age' and 'Survived': {corr_age_survived:.2f}, p-value: {p_age:.4f}")
# Interpretation: This correlation measures the relationship between passenger age ('Age') and survival ('Survived'). The correlation coefficient is -0.06, indicating a very weak negative relationship. However, the p-value (0.0528) is slightly above the typical significance level of 0.05, suggesting that the relationship between age and survival is not statistically significant. In other words, there isn't strong evidence to suggest that age has a significant impact on survival in this dataset.

# Calculate point-biserial correlation between 'Fare' and 'Survived'
corr_fare_survived, p_fare = pointbiserialr(titanic_df['Fare'], titanic_df['Survived'])
print(f"Point-biserial correlation between 'Fare' and 'Survived': {corr_fare_survived:.2f}, p-value: {p_fare:.4f}")
# Interpretation: This correlation measures the relationship between the fare paid by passengers ('Fare') and survival ('Survived'). The correlation coefficient is 0.26, indicating a moderate positive relationship. The p-value (0.0000) is very low, indicating strong evidence of a significant relationship. In simple terms, it suggests that the fare paid by passengers is positively associated with their chances of survival. Passengers who paid higher fares had a higher likelihood of survival.

# Additional Analysis: Explore the hypothesis that younger passengers survived because they paid higher fares.
# Compare the mean fare paid by survivors and non-survivors among passengers under a certain age threshold.
age_threshold = 18  # You can adjust this threshold as needed.
young_survivors = titanic_df[(titanic_df['Age'] < age_threshold) & (titanic_df['Survived'] == 1)]
young_non_survivors = titanic_df[(titanic_df['Age'] < age_threshold) & (titanic_df['Survived'] == 0)]

mean_fare_young_survivors = young_survivors['Fare'].mean()
mean_fare_young_non_survivors = young_non_survivors['Fare'].mean()

print(f"\nMean Fare for Young Survivors: {mean_fare_young_survivors:.2f}")
print(f"Mean Fare for Young Non-Survivors: {mean_fare_young_non_survivors:.2f}")
# Interpretation: This analysis compares the mean fare paid by passengers under the age of {age_threshold} who survived and those who did not. If younger passengers survived primarily because they paid higher fares, we would expect to see a significantly higher mean fare among young survivors compared to young non-survivors.



#testing older passengers like we did with younger passengers above.  Does fare have an impact on survival for older passengers?


# Define the age threshold for older passengers (e.g., 50 years)
age_threshold_old = 50

# Select older survivors and older non-survivors
older_survivors = titanic_df[(titanic_df['Age'] >= age_threshold_old) & (titanic_df['Survived'] == 1)]
older_non_survivors = titanic_df[(titanic_df['Age'] >= age_threshold_old) & (titanic_df['Survived'] == 0)]

# Calculate the mean fare for older survivors and older non-survivors
mean_fare_older_survivors = older_survivors['Fare'].mean()
mean_fare_older_non_survivors = older_non_survivors['Fare'].mean()

# Calculate the mean fare for older survivors and older non-survivors
mean_fare_older_survivors = older_survivors['Fare'].mean()
mean_fare_older_non_survivors = older_non_survivors['Fare'].mean()

print(f"Mean Fare for Older Survivors (Age >= {age_threshold_old}): {mean_fare_older_survivors:.2f}")
print(f"Mean Fare for Older Non-Survivors (Age >= {age_threshold_old}): {mean_fare_older_non_survivors:.2f}")

# Perform a t-test to determine if the difference in mean fares is significant
from scipy.stats import ttest_ind

t_stat_older, p_value_fare_age_older = ttest_ind(older_survivors['Fare'], older_non_survivors['Fare'], equal_var=False)
print(f"T-test for Fare Difference among Older Passengers (Age >= {age_threshold_old}):")
print(f"T-statistic: {t_stat_older:.2f}, p-value: {p_value_fare_age_older:.4f}")

if p_value_fare_age_older < 0.05:
    print("There is evidence to suggest a significant difference in mean fares between older survivors and older non-survivors.")
else:
    print("There is no significant evidence to suggest a difference in mean fares between older survivors and older non-survivors.")



# Perform a statistical test to determine if the difference in mean fares is significant.
from scipy.stats import ttest_ind

t_stat, p_value_fare_age = ttest_ind(young_survivors['Fare'], young_non_survivors['Fare'], equal_var=False)
print(f"T-test for Fare Difference among Young Passengers (Age < {age_threshold}):")
print(f"T-statistic: {t_stat:.2f}, p-value: {p_value_fare_age:.4f}")
if p_value_fare_age < 0.05:
    print("There is evidence to suggest a significant difference in mean fares between young survivors and young non-survivors.")
else:
    print("There is no significant evidence to suggest a difference in mean fares between young survivors and young non-survivors.")
# Interpretation: This t-test compares the mean fares of young survivors and young non-survivors. A low p-value (< 0.05) suggests that there is evidence of a significant difference in mean fares between these groups, supporting the hypothesis that younger passengers who survived paid higher fares.



# Perform logistic regression with 'Age' and 'Fare' as predictors
import statsmodels.api as sm

# Add a constant to the independent variables (intercept)
X_with_const = sm.add_constant(titanic_df[['Age', 'Fare']])
logit_model = sm.Logit(titanic_df['Survived'], X_with_const)
logit_results = logit_model.fit()

# Get the summary of the logistic regression
print(logit_results.summary())





