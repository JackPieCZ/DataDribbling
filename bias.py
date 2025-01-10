import pandas as pd

# Load the two datasets
biased_data = pd.read_csv('./data_all/elo_predictions.csv')  # ELO predictions with home team bias
unbiased_data = pd.read_csv('./data_all/elo_predictions_nobias.csv')  # ELO predictions without home team bias

# Ensure both datasets are aligned by GameID or other unique identifier
if not biased_data['GameID'].equals(unbiased_data['GameID']):
    raise ValueError("The GameID columns in the two files do not match!")

# Compare the CorrectELOPrediction column
affected_predictions = biased_data[biased_data['CorrectELOPrediction'] != unbiased_data['CorrectELOPrediction']]

# Count total affected predictions
num_affected = affected_predictions.shape[0]

# Count predictions changed to correct
corrected_to_right = affected_predictions[
    (biased_data['CorrectELOPrediction'] == True) & (unbiased_data['CorrectELOPrediction'] == False)
].shape[0]

# Count predictions changed to incorrect
corrected_to_wrong = affected_predictions[
    (biased_data['CorrectELOPrediction'] == False) & (unbiased_data['CorrectELOPrediction'] == True)
].shape[0]

# Print results
print(f"Total number of affected predictions: {num_affected}")
print(f"Number of predictions changed to correct: {corrected_to_right}")
print(f"Number of predictions changed to incorrect: {corrected_to_wrong}")