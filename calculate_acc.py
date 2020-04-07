'''
Used to calculate the accuracy of the predictions
'''
import numpy as np
import pandas as pd

# load predictions csv
predictions = pd.read_csv('data/tf_predictions_adam3.csv')
# load answer csv
answer = pd.read_csv('data/gender_submission.csv')

correct = 0
not_correct = 0

survived = predictions.to_dict(orient='PassengerId')

# for each in predictions:
    # survived = each['PassengerId'].values
    # if each['Survived'] == (answer[each['PassengerId']]):
    #     correct = correct + 1
    # else:
    # not_correct = not_correct + 1

print("acc = %.3f", (correct/(correct+not_correct)))
