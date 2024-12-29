import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import f_regression
from scipy.stats import probplot
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.api import OLS, add_constant
from mlxtend.feature_selection import SequentialFeatureSelector

def evaluate_model(model_name, y_test, y_pred, X_train):

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X_train.shape[1] - 1)

    print(f"\n{model_name} - Model Results:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Adjusted R²: {adj_r2:.4f}")

    return y_test - y_pred

def diagnostic_plots(y_pred, residuals):

    # Residuals vs Predicted Values Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Residuals vs. Predicted values")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.show()

    # Histogram of Residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title("Histogram of residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

    # Q-Q plot
    plt.figure(figsize=(10, 6))
    probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q plot")
    plt.show()

def display_coefficients_with_pvalues(model, feature_names, X, y):

    _, p_values = f_regression(X, y)

    # Combine coefficients and p-values into a DataFrame
    coefficients_info = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'P-value': p_values
    })

    # Sort by the magnitude of coefficients
    coefficients_info = coefficients_info.sort_values('Coefficient', ascending=False)

    # Display coefficients and p-values
    print("\nCoefficients and P-values:")
    print(coefficients_info)

    # Plot coefficients
    plt.figure(figsize=(12, 6))
    plt.bar(coefficients_info['Feature'], coefficients_info['Coefficient'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Coefficients of Features')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.tight_layout()
    plt.show()

def delete_outliers(X, y, threshold=4):

    # Add constant to features for OLS model (intercept)
    X_with_const = add_constant(X)
    
    # Fit OLS model
    model = OLS(y, X_with_const).fit()
    
    # Compute Cook's distance
    influence = OLSInfluence(model)
    cooks_d = influence.cooks_distance[0]
    
    # Calculate threshold for identifying outliers
    n = len(y)  # number of observations
    cook_threshold = threshold / n
    
    # Identify outliers
    outlier_indices = np.where(cooks_d > cook_threshold)[0]
    print(f"Počet odstraněných outliers: {len(outlier_indices)}")
    print(f"Indexy odstraněných bodů: {outlier_indices}")
    
    # Remove outliers from the dataset
    X_clean = np.delete(X, outlier_indices, axis=0)
    y_clean = np.delete(y, outlier_indices, axis=0)
    
    return X_clean, y_clean, outlier_indices

def calculate_vif_and_correlations(X, features, correlation_threshold=0.8):
    
    # Add constant to the features (intercept term)
    X_with_const = add_constant(X)
    
    # If X_with_const is a numpy array, convert it to a pandas DataFrame with the feature names
    if isinstance(X_with_const, np.ndarray):
        X_with_const = pd.DataFrame(X_with_const, columns=["const"] + features)
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    
    # Display VIF values
    print("VIF Values:")
    print(vif_data)
    
    # Calculate the correlation matrix
    correlation_matrix = X_with_const.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Find groups of features with high correlation
    correlated_features = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                # If correlation is above threshold, add to the list
                feature_pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])
                correlated_features.append(feature_pair)
    
    # Print groups of highly correlated features
    if correlated_features:
        print("\nHighly Correlated Features (Above Threshold):")
        for pair in correlated_features:
            print(f"{pair[0]} and {pair[1]} have correlation > {correlation_threshold}")
    else:
        print("\nNo highly correlated feature groups found above the threshold.")
    
    return vif_data, correlation_matrix, correlated_features

def remove_highly_correlated_features(X, y, features, correlated_features):

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=features)
    # Initialize a list of features to drop
    features_to_drop = []

    # Loop through the correlated feature pairs and drop one feature from each pair
    for feature1, feature2 in correlated_features:
        # Drop feature2 (you can choose to drop feature1 or make it more sophisticated)
        features_to_drop.append(feature2)

    # Drop the selected features from X
    X_clean = X.drop(columns=features_to_drop)

    # Update the features list by removing the dropped features
    remaining_features = [feature for feature in features if feature not in features_to_drop]

    # Return the cleaned X, y, and the remaining features
    return X_clean, y, remaining_features

def linear_regression(X, y, features):

    print("LINEAR REGRESSION")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lin_reg.predict(X_test)


    residuals = evaluate_model("Linear Regression", y_test, y_pred, X_train)
    display_coefficients_with_pvalues(lin_reg, features, X, y)
    diagnostic_plots(y_pred, residuals)

def polynomial_regression(X, y, features):

    print("POLYNOMIAL REGRESSION")

    # Transform features into polynomial features (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Get the new feature names after transformation
    poly_features = poly.get_feature_names_out(features)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Initialize and fit the Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lin_reg.predict(X_test)

    residuals = evaluate_model("Polynomial Regression", y_test, y_pred, X_train)
    display_coefficients_with_pvalues(lin_reg, poly_features, X_poly, y)
    diagnostic_plots(y_pred, residuals)

def lasso_regression(X, y, features):
   
    print("LASSO REGRESSION")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit LassoCV (with 5-fold cross-validation to find the best alpha)
    lasso_cv = LassoCV(cv=5, random_state=42)
    lasso_cv.fit(X_train, y_train)

    # Get the best alpha from cross-validation
    best_alpha = lasso_cv.alpha_
    print(f"Optimal value of alpha: {best_alpha}")

    # Make predictions on the test set
    y_pred = lasso_cv.predict(X_test)

    residuals = evaluate_model("Lasso Regression", y_test, y_pred, X_train)
    display_coefficients_with_pvalues(lasso_cv, features, X, y)
    diagnostic_plots(y_pred, residuals)

def ridge_regression(X, y, features):

    print("RIDGE REGRESSION")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit RidgeCV (with 5-fold cross-validation to find the best alpha)
    ridge_cv = RidgeCV(cv=5)
    ridge_cv.fit(X_train, y_train)

    # Get the best alpha from cross-validation
    best_alpha = ridge_cv.alpha_
    print(f"Optimal value of alpha: {best_alpha}")

    # Make predictions on the test set
    y_pred = ridge_cv.predict(X_test)

    residuals = evaluate_model("Ridge Regression", y_test, y_pred, X_train)
    display_coefficients_with_pvalues(ridge_cv, features, X, y)
    diagnostic_plots(y_pred, residuals)

def forward_stepwise_selection(X, y, features):

    print("FORWARD STEPWISE SELECTION")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the model
    model = LinearRegression()
    
    # Create the Forward Stepwise Selection object
    sfs_forward = SequentialFeatureSelector(model, 
                                            k_features="best",  # Select the best number of features
                                            forward=True,        # Forward Selection
                                            floating=False,      # Without floating version (no removal and re-adding)
                                            scoring="neg_mean_squared_error",  # Minimize MSE
                                            cv=5)                # 5-fold cross-validation
    
    # Fit the model to the training data
    sfs_forward.fit(X_train, y_train)
    
    # Get selected features
    selected_features_forward = list(sfs_forward.k_feature_idx_)
    print("\nFeatures selected by Forward Stepwise Selection:")
    selected_features_names_forward = [features[i] for i in selected_features_forward]
    print(selected_features_names_forward)
    
    # Get unselected features
    unselected_features_forward = list(set(features) - set(selected_features_names_forward))
    print("\nFeatures NOT selected by Forward Stepwise Selection:")
    print(unselected_features_forward)
    
    # Evaluate the model on selected features
    X_selected_forward = X_train.iloc[:, selected_features_forward]
    model_forward = LinearRegression()
    model_forward.fit(X_selected_forward, y_train)
    
    # Predict on the test data
    X_test_selected_forward = X_test.iloc[:, selected_features_forward]  # Adjust indexing for test set
    y_pred_forward = model_forward.predict(X_test_selected_forward)

    residuals_forward = evaluate_model("Forward Stepwise Selection", y_test, y_pred_forward, X_train)
    diagnostic_plots(y_pred_forward, residuals_forward)
    display_coefficients_with_pvalues(model_forward, selected_features_names_forward, X_selected_forward, y_train)

def backward_stepwise_selection(X, y, features):

    print("BACKWARD STEPWISE SELECTION")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the model
    model = LinearRegression()
    
    # Create the Backward Stepwise Selection object
    sfs_backward = SequentialFeatureSelector(model, 
                                             k_features="best",  # Select the best number of features
                                             forward=False,       # Backward Selection
                                             floating=False,      # Without floating version
                                             scoring="neg_mean_squared_error",  # Minimize MSE
                                             cv=5)                # 5-fold cross-validation
    
    # Fit the model to the training data
    sfs_backward.fit(X_train, y_train)
    
    # Get selected features
    selected_features_backward = list(sfs_backward.k_feature_idx_)
    print("\nFeatures selected by Backward Stepwise Selection:")
    selected_features_names_backward = [features[i] for i in selected_features_backward]
    print(selected_features_names_backward)
    
    # Get unselected features
    unselected_features_backward = list(set(features) - set(selected_features_names_backward))
    print("\nFeatures NOT selected by Backward Stepwise Selection:")
    print(unselected_features_backward)
    
    # Evaluate the model on selected features
    X_selected_backward = X_train.iloc[:, selected_features_backward]
    model_backward = LinearRegression()
    model_backward.fit(X_selected_backward, y_train)
    
    # Predict on the test data
    X_test_selected_backward = X_test.iloc[:, selected_features_backward]  # Adjust indexing for test set
    y_pred_backward = model_backward.predict(X_test_selected_backward)

    residuals_backward = evaluate_model("Backward Stepwise Selection", y_test, y_pred_backward, X_train)
    diagnostic_plots(y_pred_backward, residuals_backward)
    display_coefficients_with_pvalues(model_backward, selected_features_names_backward, X_selected_backward, y_train)

# Load data
games = pd.read_csv('./testing/data/games.csv')
#players = pd.read_csv('./testing/data/players.csv')

# Features
features_game = ['H', 'A', 'N', 'POFF', 'HFGA', 'AFGA', 'HFG3M', 'AFG3M',
            'HFG3A', 'AFG3A', 'HFTM', 'AFTM', 'HFTA', 'AFTA', 'HORB', 'AORB',
            'HDRB', 'ADRB', 'HRB', 'ARB', 'HAST', 'AAST', 'HSTL', 'ASTL',
            'HBLK', 'ABLK', 'HTOV', 'ATOV', 'HPF', 'APF'] # deleted HFGM, AFGM, HSC, ASC

X = games[features_game]
y = games['HSC'] - games['ASC']  # score difference

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# first linear regression model
linear_regression(X_scaled, y, features_game) 

# data cleaning
X_0, y_0, outliers = delete_outliers(X_scaled, y)
# find multicollinearity
vif_data, correlation_matrix, correlated_features = calculate_vif_and_correlations(X_0, features_game)
# remove features with high multicollinearity
X_clean, y_clean, remaining_features = remove_highly_correlated_features(X_0, y_0, features_game, correlated_features)

linear_regression(X_clean, y_clean, remaining_features)
polynomial_regression(X_clean, y_clean, remaining_features)
lasso_regression(X_clean, y_clean, remaining_features)
ridge_regression(X_clean, y_clean, remaining_features)
forward_stepwise_selection(X_clean, y_clean, remaining_features)
backward_stepwise_selection(X_clean, y_clean, remaining_features)
