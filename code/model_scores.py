'''
compute classification accuracy and R^2 scores together with the 95%
confidence interval
'''
from sklearn.metrics import accuracy_score
from confint import classification_confint, regression_confint

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

def model_r2(model, X, y):
  '''
  compute R^2 score together with the 95% confidence interval
  Parameters:
    model - estimator
    X - independent features
    y - target vector
  Returns:
    (r2,lb,ub) - R^2, lowerbound, upperbound
  '''
  # compute the R^2 score of regression model      
  r2 = model.score(X,y)

  # 95% confidence interval
  (lb,ub) = regression_confint(r2,X.shape[0],X.shape[1])

  # return a triple
  return (r2,lb,ub)

