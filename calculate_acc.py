'''
Used to calculate the accuracy of the predictions
'''
import numpy as np
import pandas as pd
import sklearn
# load predictions csv
predictions = pd.read_csv('data/predictions/tf_predictions_name.csv')
# load answer csv
answer = pd.read_csv('data/predictions/submission.csv')


predictions_np = predictions.values
answer_np = answer.values

correct = 0
not_correct = 0

for idx in range(len(predictions_np)):
    a = predictions_np[idx][1]
    b = answer_np[idx][1]
    if (a == b):
        correct = correct + 1
    else:
        not_correct = not_correct + 1
        
    
print("%.5f"%(correct/(not_correct+correct)))