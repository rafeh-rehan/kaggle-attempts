# Preamble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Turn off warnings
import warnings
warnings.filterwarnings("ignore")

## ML stuff

#XGBClassifier for our model
from xgboost import XGBClassifier

# Data Processing tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load data
train = pd.read_csv('./train.csv')
test  = pd.read_csv('./test.csv')

# %% [code]
# Take a quick peek
train.shape
train.head()

# Column exploration
train['Age'].plot(kind='hist')
train['Parch'].value_counts()
train['Ticket'].value_counts()
train['Embarked'].value_counts()

# Select Independent and Target variables
x_train_full = train.drop(['PassengerId','Name'], axis=1)
y_train_full = x_train_full.pop('Survived')


# Splitting train data into validation and training subsets
x_train, x_test, y_train, y_test = train_test_split(x_train_full,y_train_full,
                                                    test_size=0.3,random_state=0)

# %% [code]
# Independent variables with missing values
num_cols = ['Pclass','Age','SibSp','Parch','Fare']
cat_cols = ['Sex','Ticket','Cabin','Embarked']
# x_train.columns

# Numerical imputer to replace missing numerical age data
num_transformer = Pipeline(steps=[ ('num_imp', SimpleImputer(strategy='median')),
                                   ('scalar', StandardScaler() ) ] )

# Imputer and one hot encoder for categorical data
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('onehot', OneHotEncoder(sparse=False, handle_unknown="ignore"))])

# Bundle both transformers into a single preprocessing step
preprocessor = ColumnTransformer(transformers=[('num',num_transformer, num_cols),
                                               ('cat',cat_transformer, cat_cols)],
                                                n_jobs=-1)

# x_train.head()
# Preprocess training and validation data from test file
x_train = preprocessor.fit_transform(x_train)
x_test  = preprocessor.transform(x_test)

# %% [code]
# Establish our model, and find parameters which minimize errors via grid search
def xgb_search(n_start,n_end,dd,l_start,l_end,dd2):
#
#  Uses a rudimentary 2D grid search to determine best parameter values for
#  XGBRegressor.
#
#  Parameters:
#              dd  - density of n_estimators grid search
#              dd2 - density of learning_rate grid search
#
    score_best = 0.0  # Initialize best score
    n_spacing = np.linspace(n_start, n_end, dd)
    l_spacing = np.linspace(l_start, l_end, dd2)
    for i in range(len(n_spacing)):         # n_estimators search
        n = int(n_spacing[i])
        print(n)
        for j in range(len(l_spacing)):     # learning_rate
            l = l_spacing[j]
            print(l)

            # Create our model for each grid node
            model = XGBClassifier(n_estimators=n, learning_rate=l, random_state=0,
                                     objective='binary:hinge',n_jobs=8)

            # Create our pipeline for preprocessing and estimating
            my_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

            # Evaluate model through cross validation of entire training set
            score = cross_val_score(estimator=my_pipe, X=x_train_full, y=y_train_full,
                            cv=3, n_jobs=-1, error_score='raise', scoring='accuracy')

            # Store params for best scores
            score = score.mean()
            print('score',score)
            score_best = max(score,score_best)
            if score_best == score:
                n_estimators_best = n
                learning_rate_best = l

    return score_best, n_estimators_best, learning_rate_best

# Perform a grid search for optimal parameters
n_start = 100
n_end   = 1000
dd      = 10

l_start = 0.001
l_end   = 0.05
dd2     = 10

# Grid search
# score_best, n_best, l_best = xgb_search(n_start,n_end,dd,l_start,l_end,dd2)

# Store best parameters
score_best = 0.8226711560044894
n_best     = 400
l_best     = 0.017333333333333333

# Print results
print("Best score:",score_best,"\nOptimal n_estimators:",n_best,
"\nOptimal learning_rate:",l_best)

# %% [code]
# Load optimal parameters and create a pipeline
best_model = XGBClassifier(n_estimators=n_best,learning_rate=l_best,random_state=0,
                              objective='binary:hinge',n_jobs=-1)

best_pipe  = Pipeline(steps=[('preprocessor',preprocessor),
                             ('model',best_model)])

# Double check accuracy
score = cross_val_score(estimator=best_pipe, X=x_train_full, y=y_train_full,
                cv=4, n_jobs=8, error_score='raise', scoring='accuracy')
print(score,abs(score).mean())


# More accuracy tests
history = best_model.fit(x_train, y_train)
preds = best_model.predict(x_test)
err = accuracy_score(y_test,preds)

print("Test Accuracy:",err)

# %% [code]
# Prediction of test file
X = test.drop(['PassengerId','Name'], axis=1)

final_model = best_pipe.fit(x_train_full,y_train_full)
preds = final_model.predict(X)

# Take a peek at the predictions
print("Predictions for competition:", preds)

# Create submission file
submission = pd.DataFrame(preds,index=test['PassengerId'], columns = ['Survived'])
submission.to_csv('predictions_main1.csv')
