import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load datasets
games = pd.read_csv('./data/games.csv')
players = pd.read_csv('./data/players.csv')

# Feature Engineering and Aggregation
# Aggregate player statistics to team level per game
# player_agg = players.groupby(['Season', 'Date', 'Team']).agg({
#     'MIN': 'sum',
#     'FGM': 'sum',
#     'FGA': 'sum',
#     'FG3M': 'sum',
#     'FG3A': 'sum',
#     'FTM': 'sum',
#     'FTA': 'sum',
#     'ORB': 'sum',
#     'DRB': 'sum',
#     'RB': 'sum',
#     'AST': 'sum',
#     'STL': 'sum',
#     'BLK': 'sum',
#     'TOV': 'sum',
#     'PF': 'sum',
#     'PTS': 'sum'
# }).reset_index()
# print(f"{games.shape=}")
# # Merge aggregated player stats back to games data
# games = games.merge(player_agg, left_on=['Season', 'Date', 'HID'], right_on=[
#                     'Season', 'Date', 'Team'], suffixes=('', '_H'))
# games = games.merge(player_agg, left_on=['Season', 'Date', 'AID'], right_on=[
#                     'Season', 'Date', 'Team'], suffixes=('', '_A'))
# print(f"{games.shape=}")

# Drop redundant columns if they exist
for col in ['Team_H', 'Team_A']:
    if col in games.columns:
        games = games.drop(columns=[col])

# Define target variable (1 if Home team wins, 0 otherwise)
# games['target'] = games['H']

# Feature Selection - Selecting key features for modeling
features = [
    'H', 'A',
    'HSC', 'ASC',
    'N', 'POFF', 'HFGM', 'AFGM', 'HFGA', 'AFGA', 'HFG3M', 'AFG3M',
    'HFG3A', 'AFG3A', 'HFTM', 'AFTM', 'HFTA', 'AFTA', 'HORB', 'AORB',
    'HDRB', 'ADRB', 'HRB', 'ARB', 'HAST', 'AAST', 'HSTL', 'ASTL',
    'HBLK', 'ABLK', 'HTOV', 'ATOV', 'HPF', 'APF',
    # 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
    # 'MIN_A', 'FGM_A', 'FGA_A', 'FG3M_A', 'FG3A_A', 'FTM_A', 'FTA_A', 'ORB_A', 'DRB_A', 'RB_A', 'AST_A', 'STL_A', 'BLK_A', 'TOV_A', 'PF_A', 'PTS_A'
]

X = games[features]
y = 1 / games['OddsH']
print(f"{X.shape=}")
print(f"{y.shape=}")
print(f"{X.columns=}")
print(f"{X.head()=}")
print(f"{y.head()=}")
# input("Press Enter to continue...")
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model for continuous predictions
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Make predictions
y_pred = linear_reg.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# Print and visualize feature coefficients
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': linear_reg.coef_
})
feature_importance = feature_importance.sort_values('Coefficient', ascending=False)

print("\nFeature Coefficients:")
print(feature_importance)

# Visualize coefficients with a bar plot
plt.figure(figsize=(12, 6))
plt.bar(feature_importance['Feature'], feature_importance['Coefficient'])
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance based on Linear Regression Coefficients')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.tight_layout()
plt.show()

# Calculate feature importance using mutual information
from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X_train, y_train)
mi_importance = pd.DataFrame({
    'Feature': features,
    'Mutual Information Score': mi_scores
}).sort_values('Mutual Information Score', ascending=False)

print("\nFeature Importance using Mutual Information:")
print(mi_importance)

plt.figure(figsize=(12, 6))
plt.bar(mi_importance['Feature'], mi_importance['Mutual Information Score'])
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance based on Mutual Information')
plt.xlabel('Features')
plt.ylabel('Mutual Information Score')
plt.tight_layout()
plt.show()


# Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict probabilities for the test set
# y_pred_proba = logreg.predict(X_test)[:, 1]

# Evaluate model performance
accuracy = accuracy_score(y_test, logreg.predict(X_test))
# roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Accuracy: {accuracy:.2f}')
# print(f'ROC AUC: {roc_auc:.2f}')

# Feature Importance using Permutation Importance
perm_importance = permutation_importance(logreg, X_test, y_test, n_repeats=10, random_state=42)
feature_importances = pd.Series(perm_importance.importances_mean, index=features)

# Plotting Feature Importances
feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(15, 7))
plt.title("Feature Importance based on Permutation Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# Calculate probabilities of Home and Away win for use with Kelly's criterion
# Adding predictions back to the original dataset for betting purposes
games['home_win_prob'] = logreg.predict_proba(X)[:, 1]
games['away_win_prob'] = 1 - games['home_win_prob']

# Display a few rows to understand the results
print(games[['Season', 'Date', 'HID', 'AID', 'home_win_prob', 'away_win_prob']].head())
