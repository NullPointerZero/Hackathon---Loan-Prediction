

# At first import all the modules that are needed:
import pandas as pd 
import matplotlib as plt
import numpy as np
from  sklearn.linear_model import Ridge

# import the test data aswell as the train data 
train = pd.read_csv("train_ctrUa4K.csv")

train_cleaned = train.dropna()


print(train_cleaned[train_cleaned["Loan_Status"]=="Y"].sort_values(["LoanAmount"],ascending=True).head(10))
