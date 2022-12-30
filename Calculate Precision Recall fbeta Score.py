from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score

def calculateMetric(y_true, y_pred, b):
  precision = round(precision_score(y_true, y_pred), 2)
  recall = round(recall_score(y_true, y_pred), 2)
  f = round(fbeta_score(y_true, y_pred, beta = b), 2)
  
  metric = [precision, recall, f]
  return metric

y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

precision, recall, fbeta = calculateMetric(y_true, y_pred, 0.5)
print("Precision Score =", precision)
print("Recall Score = ", recall)
print("F-beta Score = ", fbeta)