

# At first import all the modules that are needed:
import pandas as pd 
import matplotlib as plt
import numpy as np
from  sklearn.neighbors import KNeighborsClassifier


# import the test data aswell as the train data 
train = pd.read_csv("train_ctrUa4K.csv")
test = pd.read_csv("test_lAUu6dG.csv")
submission_df = pd.read_csv("sample_submission_file.csv")

train_cleaned = train.fillna(0.5)
test_cleaned = test.fillna(0.5)

# Lets select some features that might be needed for the evaluation:


features = ["Gender","Married","Dependents","Education","ApplicantIncome","LoanAmount","Credit_History", "Loan_Status"]
test_features = ["Gender","Married","Dependents","Education","ApplicantIncome","LoanAmount","Credit_History"]
train_cleaned = train_cleaned[features]
test_cleaned = test_cleaned[test_features]
# following should be quantisized Gender: Male/Female, Education: Graduated/Ungraduated, Married: Yes/No, Loan_Status Y/N
# Then make everything a floating number
# Then normalize everything from 0 to 1

# Quantizising:
train_cleaned.loc[train_cleaned["Gender"] == "Male","Gender"] = 1.0
train_cleaned.loc[train_cleaned["Gender"] == "Female","Gender"] = 0.0
test_cleaned.loc[test_cleaned["Gender"] == "Male", "Gender"] = 1.0
test_cleaned.loc[test_cleaned["Gender"] == "Female", "Gender"] = 0.0

train_cleaned.loc[train_cleaned["Married"] == "Yes", "Married"] = 1.0
train_cleaned.loc[train_cleaned["Married"] == "No", "Married"] = 0.0
test_cleaned.loc[test_cleaned["Married"] == "Yes", "Married"] = 1.0
test_cleaned.loc[test_cleaned["Married"] == "No", "Married"] = 0.0

train_cleaned.loc[train_cleaned["Loan_Status"] == "Y", "Loan_Status"] = 1.0
train_cleaned.loc[train_cleaned["Loan_Status"] == "N", "Loan_Status"] = 0.0

train_cleaned.loc[train_cleaned["Dependents"] == "3+", "Dependents"] = 3.0
test_cleaned.loc[test_cleaned["Dependents"] == "3+", "Dependents"] = 3.0


train_cleaned.loc[train_cleaned["Education"] == "Graduate", "Education"] = 1.0
train_cleaned.loc[train_cleaned["Education"] == "Not Graduate", "Education"] = 0.0
test_cleaned.loc[test_cleaned["Education"] == "Graduate", "Education"] = 1.0
test_cleaned.loc[test_cleaned["Education"] == "Not Graduate", "Education"] = 0.0

# Floating:
train_cleaned = train_cleaned.astype(float)
test_cleaned = test_cleaned.astype(float)

# normalize
train_cleaned = (train_cleaned - train_cleaned.min()) / (train_cleaned.max() - train_cleaned.min())
test_cleaned = (test_cleaned - test_cleaned.min()) / (test_cleaned.max() - test_cleaned.min())

#print(train_cleaned["Gender"].value_counts())

def knc_train(train_list, target_col, train_df, test_df):
    knc = KNeighborsClassifier()
    
    #np.random.seed(42)
    #shuffled_index = np.random.permutation(df.index)
    #shuffled_df = df.reindex(shuffled_index)
    
    # not neccessary anymore - split to train and testdata
    #last_train_row = int(len(shuffled_df) / 2)
    #train_df = shuffled_df.iloc[0:last_train_row]
    #test_df = shuffled_df.iloc[last_train_row:]

    # Fit the model
    knc.fit(train_df[train_list], train_df[target_col])

    # Make Predictions
    predicted_labels = knc.predict(test_df[train_list])
    return predicted_labels

training_list = ["Gender", "Education", "ApplicantIncome","LoanAmount", "Credit_History"]
predicted_labels = knc_train(training_list, "Loan_Status", train_cleaned, test_cleaned)

test_cleaned["predicted"] = predicted_labels

loan_bool = test_cleaned["predicted"] == 1.0
submission_df.loc[loan_bool,"Loan_Status"] = "Y"

submission_df.to_csv("my_submission.csv", index = False)

