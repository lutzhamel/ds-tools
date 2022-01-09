'''
compute classification accuracy and R^2 scores together with the 95%
confidence interval
'''
from sklearn.metrics import accuracy_score
from confint import classification_confint

def model_accuracy(model, X, y):
  '''
  compute classification accuracy together with the 95% confidence interval
  Parameters:
    model - estimator
    X - independent features
    y - target vector
  Returns:
    (acc,lb,ub) - accuracy, lowerbound, upperbound
  '''
  # compute the accuracy of optimal classifier      
  predict_y = model.predict(X)
  acc = accuracy_score(y, predict_y)

  # 95% confidence interval
  (lb,ub) = classification_confint(acc,X.shape[0])

  # return a triple
  return (acc,lb,ub)
