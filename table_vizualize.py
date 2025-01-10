import pandas as pd

# Load the CSV file
data = pd.read_csv('./data_all/elo_predictions.csv') 

# Calculate the four categories
both_correct = data[(data['CorrectELOPrediction'] == True) & (data['CorrectOddsPrediction'] == True)].shape[0]
elo_only = data[(data['CorrectELOPrediction'] == True) & (data['CorrectOddsPrediction'] == False)].shape[0]
odds_only = data[(data['CorrectELOPrediction'] == False) & (data['CorrectOddsPrediction'] == True)].shape[0]
neither_correct = data[(data['CorrectELOPrediction'] == False) & (data['CorrectOddsPrediction'] == False)].shape[0]

# Create a 2x2 table
table = pd.DataFrame(
    {
        "Odds Prediction": ["Correct", "Incorrect"],
        "Correct": [both_correct, elo_only],
        "Incorrect": [odds_only, neither_correct],
    },
    index=["ELO Prediction: Correct", "ELO Prediction: Incorrect"]
)

# Display the table
print(table)

# Optional: Save the table to a CSV file
table.to_csv('./prediction_results_table.csv', index=True)