'''
Used to calculate the accuracy of the predictions
'''
import numpy as np
import pandas as pd
import sklearn

# # load predictions csv
# predictions = pd.read_csv('data/predictions/stacking_test1.csv')
# # load answer csv
# answer = pd.read_csv('data/predictions/submission.csv')

def acc(predictions_np, answer_np):
    correct = 0
    not_correct = 0
    for idx in range(len(predictions_np)):
        a = predictions_np[idx]
        b = answer_np[idx]
        if (a == b):
            correct = correct + 1
        else:
            not_correct = not_correct + 1
    return correct/(not_correct+correct)

def calculate_acc(predictions_np):
    # load answer csv
    answer = pd.read_csv('data/predictions/submission.csv')
    answer_np = answer['Survived'].values
    print("acc = %.5f"%(acc(predictions_np, answer_np)))