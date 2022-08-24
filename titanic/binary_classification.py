# Preamble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Turn off warnings
import warnings
warnings.filterwarnings("ignore")

## ML stuff

#Models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Data Processing tools
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load data
train = pd.read_csv('./train.csv')
test  = pd.read_csv('./test.csv')

# Take a quick peek
train.shape
train.head()

# Column exploration
train['Age'].plot(kind='hist')
train['Cabin'].value_counts()
train['Ticket'].value_counts()
train['Embarked'].value_counts()

train['Survived'].value_counts() # pretty balanced around 60:40 split

# Find number of na's in each column
def num_na(df):
    na_series = pd.Series()
    for i in df.columns:
        na_series[i] = df[i].isna().sum()
    return na_series

nas_train = num_na(train)
nas_test  = num_na(test)
print(nas_train, nas_test)
test.shape
# Cabin is missing a LOT of data in both train and test set. Imputing the values
# wont be that helpful since the majority of the column is NA. Drop column later

# Feature selection and correlation analysis
data = train.copy()
data = data.drop(['PassengerId','Name','Cabin'], axis=1)
print(data.columns)

# Columns we're keeping for processing
num_cols = ['Pclass','Age','SibSp','Parch','Fare']
cat_cols = ['Sex','Embarked','Ticket']#['Sex','Ticket','Embarked']

# Label encode categorical data
for cols in cat_cols:
    data[cols], _ = data[cols].factorize()

# Check Na's before scoring mutual info
nas_data = num_na(data)
print(nas_data)

# Age has some na values so impute them by the average age
data_imp = SimpleImputer(strategy='mean')
data_array = data_imp.fit_transform(data)
data2 = pd.DataFrame(data_array, index=data.index, columns=data.columns)

nas_data2 = num_na(data2)
print(nas_data2) # no missing values, now we can proceed

correlation = data2.corr() # Correlation between numerical variables
plt.figure(1, figsize=(9,9))
plt.title('Correlation Map')
sns.heatmap(correlation, annot=True, cmap='Blues')
# Target (Survived) most correlated with Fare, Pclass and Sex

# Compute Mutual information
data2_y = data2.pop('Survived')
mi_scores = mutual_info_classif(data2, data2_y, discrete_features=discrete_feats)
mi_scores = pd.Series(mi_scores, index=data2.columns, name='Mutual info classification')
print(mi_scores)
# Sex and Fare still show mutual info with Survived as expected. So does Ticket

# Select Independent and Target variables (For now keep Ticket data)
x_train_full = train.drop(['PassengerId','Name','Cabin'], axis=1)
y_train_full = x_train_full.pop('Survived')

print(num_na(x_train_full)) # Some missing data

# Numerical imputer to replace missing numerical age data
num_transformer = Pipeline(steps=[ ('num_imp', SimpleImputer(strategy='mean')),
                                   ('scalar', StandardScaler() ) ] )

# Imputer and one hot encoder for categorical data
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('onehot', OneHotEncoder(sparse=False, handle_unknown="ignore"))])

# Bundle both transformers into a single preprocessing step
preprocessor = ColumnTransformer(transformers=[('num',num_transformer, num_cols),
                                               ('cat',cat_transformer, cat_cols)],
                                                n_jobs=-1)

# Establish an optimal xgb model
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
                                     objective='binary:logistic',n_jobs=-1, early_stopping_rounds=10)

            # Create our pipeline for preprocessing and estimating
            my_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

            # Stratified k fold splitting
            kfolds = StratifiedKFold(n_splits = 4, shuffle=True, random_state=0)
            # Evaluate model through cross validation of entire training set
            score = cross_val_score(estimator=my_pipe, X=x_train_full, y=y_train_full,
                            cv=kfolds, n_jobs=-1, error_score='raise', scoring='accuracy')

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

# Store best parameters (binary:hinge)
# score_best = 0.8260381593714926
# n_best     = 100
# l_best     = 0.037750000000000006

# Store best parameters (binary:hinge)
score_best = 0.8327879448955682
n_best     = 400
l_best     = 0.0064444444444444445

# Print results
print("Best score:",score_best,"\nOptimal n_estimators:",n_best,
"\nOptimal learning_rate:",l_best)

# Load optimal parameters for xgb model
xgb = XGBClassifier(n_estimators=n_best,learning_rate=l_best,random_state=0,
                              objective='binary:logistic',n_jobs=-1,
                              early_stopping_rounds=10)

# Create several other models for testing
logistic = LogisticRegression()
svc = SVC()

models = [xgb, logistic, svc]

### Find which model works best for binary classification
# First, define scoring function to pass to cross_val_score
def my_scorer(y_true, y_pred):
    print(classification_report(y_true, y_pred)) # print classification report
    print(confusion_matrix(y_true, y_pred)) # print confusion matrix
    print('test')
    return accuracy_score(y_true, y_pred) # return accuracy score

# Test various models with our defined scoring method
def model_metrics(models, scoring_func, X, y, cv):
# Tests any number of models with a given scoring metric
# through cross validation
#
# models       - List of models from scikit learn
# scoring_func - Scoring method to pass to cross_val_score
    kfolds = StratifiedKFold(n_splits = cv, shuffle=True, random_state=0)
    for model in models:
        print(str(model)) # print which model is being tested
        # Create pipeline
        pipe  = Pipeline(steps=[('preprocessor',preprocessor),
                        ('model', model)])
        # Evaluate model through cross validation of entire training set
        score = cross_val_score(estimator=pipe, X=x_train_full, y=y_train_full,
                        cv=kfolds, n_jobs=-1, error_score='raise',
                        scoring = make_scorer(scoring_func))
        print(score, score.mean())

# Determine performance of each model
model_metrics(models, my_scorer, x_train_full, y_train_full, cv = 4)
# SVC and xgbclassif seem to work best, XGB slightly better
# Explore these options a little further

# Splitting train data into validation and training subsets
x_train, x_test, y_train, y_test = train_test_split(x_train_full,y_train_full,
                                                    test_size=0.33,random_state=0)

x_train = preprocessor.fit_transform(x_train)
x_test  = preprocessor.transform(x_test)

# Closer look at Xgb vs SVC
svc = svc.fit(x_train, y_train)
svc_preds = svc.predict(x_test)
svc_acc   = accuracy_score(y_test, svc_preds)

xgb = xgb.fit(x_train,y_train)
xgb_preds = xgb.predict(x_test)
xgb_acc   = accuracy_score(y_test, xgb_preds)

print("Xgb accuracy:", xgb_acc, '\nSVC accuracy:', svc_acc)
# XGB better it seems

# The real deal
X = test.drop(['PassengerId','Name','Cabin'], axis=1)

# Process training data and test data
X_train = preprocessor.fit_transform(x_train_full)
X       = preprocessor.transform(X)
Y_train = y_train_full

# fit and predict using our best model
final_model = xgb.fit(X_train, Y_train)
preds = final_model.predict(X)

# Take a peek at the predictions
print("Predictions for competition:", preds)

submission = pd.DataFrame(preds,index=test['PassengerId'], columns = ['Survived'])
submission.to_csv('predictions_main1.csv')
