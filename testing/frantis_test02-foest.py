import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def adjust_features_for_team22(team_22_matches):
    """
    Adjusts features in the team_22_matches DataFrame such that all relevant columns
    (HID, HSC, etc.) correspond to "home" if they represent team 22 and "alien" otherwise.

    Parameters:
        team_22_matches (pd.DataFrame): Input DataFrame containing team 22 matches.

    Returns:
        pd.DataFrame: Modified DataFrame with adjusted features.
    """
    team_22_matches_modified = team_22_matches.copy()

    # Identify home and alien columns
    home_cols = [col for col in team_22_matches.columns if col.startswith('H')]
    alien_cols = [col for col in team_22_matches.columns if col.startswith('A')]

    # Adjust data based on whether team 22 is home or alien
    is_team_22_home = team_22_matches["HID"] == 22

    for home_col, alien_col in zip(home_cols, alien_cols):
        # Swap values for matches where team 22 is not home
        team_22_matches_modified[home_col] = np.where(
            is_team_22_home,
            team_22_matches[home_col],
            team_22_matches[alien_col]
        )
        team_22_matches_modified[alien_col] = np.where(
            is_team_22_home,
            team_22_matches[alien_col],
            team_22_matches[home_col]
        )

    # Rename columns to standardize the representation
    """
    renamed_cols = {col: col[1:] for col in home_cols + alien_cols}
    team_22_matches_modified.rename(columns=renamed_cols, inplace=True)
    """

    return team_22_matches_modified


# Sample data /testing/data/team_22_matches.csv
team_22_matches_d = pd.read_csv('./testing/data/team_22_matches.csv')

# Create DataFrame
team_22_matches = pd.DataFrame(team_22_matches_d)

team_22_matches_modified = adjust_features_for_team22(team_22_matches)

# TODO rate features for linear model describing whether team 22 won

# Select features for the model
# Use older data to get result od upcoming match


# features = ["HSC", "HTOV", "HAST", "HFGM", "HPF", "HBLK"]
features = ["H", "HSC", "HFGM", "HFGA", "HFG3M", "HFG3A", "HFTM",
            "HFTA", "HORB", "HDRB", "HRB", "HAST", "HSTL", "HBLK", "HTOV", "HPF"]
X = team_22_matches_modified[features].iloc[:-21]
y = team_22_matches_modified["Team22_Won"].iloc[1:-20]
"""
X = team_22_matches[features].iloc[:, :-1]
y = team_22_matches_modified["Team22_Won"].iloc[1:]
"""
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Feature importance
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

print("Feature Importances:")
print(importance_df)

# Select top features
top_features = importance_df["Feature"][:6]  # Adjust number of top features as needed
X_train_top = X_train[:, :6]
X_test_top = X_test[:, :6]

# Retrain model with selected features
rf_model_top = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model_top.fit(X_train_top, y_train)

# Evaluate the model
y_pred_prob = rf_model_top.predict_proba(X_test_top)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nROC AUC Score with Selected Features: {roc_auc:.2f}")

# Visualization of Feature Importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importance from Random Forest")
plt.gca().invert_yaxis()
plt.show()
