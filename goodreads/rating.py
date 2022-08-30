# Preamble
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
tf.config.run_functions_eagerly(True)

import tensorflow_hub as hub
import tensorflow_text

# Turn off warnings
import warnings
warnings.filterwarnings("ignore")

## ML stuff

# Data Processing Tools and metrics
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, FeatureAgglomeration

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Language Processing
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Meta modelling
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.svm import LinearSVC

# SGD apparently works well for text data
from sklearn.linear_model import SGDClassifier

# Decision Trees
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier

#Neural Networks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM
from tensorflow.keras.layers import BatchNormalization, TextVectorization
from tensorflow.keras.layers import Embedding, Input, StringLookup

# Wrapper to make neural network compitable with StackingRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Run tensorflow on CPU
tf.device('/cpu:0')

# Load training and test data
train_full = pd.read_csv('goodreads_train.csv')
test  = pd.read_csv('goodreads_test.csv')

# Data too much fr fr
train = train_full.sample(frac=0.1)
train.shape


# Convert
# train.dropna(axis=0)
# train['read_at'] = train['read_at'].apply(lambda x: datetime.strptime(str(x), '%a %b %d %H:%M:%S %z %Y'))


#Drop data with missing values
X = train.dropna(axis=0)
X = X.drop(columns=['review_id'])
y = X.pop('rating')

k = X.to_numpy()
# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)


### preprocessing
# column dtypes
num_cols = [cols for cols in X.columns if X[cols].dtypes == (type(1) or type(1.0))]
cat_cols = [cols for cols in X.columns if X[cols].dtypes == type(object)]


# Start with text data only, convert to string arrays
import re
from time import time

def string_cleanup(df, str_cols):
# Clean string/object columns from pandas DataFrame.
# Returns new dataframe of only cleaned text.

    # Grab string/object columns from data frame
    df_text = df[str_cols].to_numpy()
    xt = df_text.transpose()

    # Clean strings, stem and stop words
    stemmer = nltk.stem.WordNetLemmatizer()
    t1 = time()
    for i in range(len(xt)): # For each column
        # Grab strings
        train_strings = xt[i]
        for j in range(len(train_strings)): # For each string in the column
            # Lower case all letters first
            train_strings[j] = train_strings[j].lower()

            # Remove special characters and punctuation
            train_strings[j] = re.sub(r'\W', ' ', train_strings[j])
            train_strings[j] = re.sub(r'[^\w\s]',' ',train_strings[j])

            # Remove all single characters
            train_strings[j] = re.sub(r'\s+[a-zA-Z]\s+', ' ', train_strings[j])

            # Remove extra spaces
            train_strings[j] = re.sub(r'\s+', ' ', train_strings[j], flags=re.I)

            # Lematize
            train_strings[j] = train_strings[j].split()
            train_strings[j] = [stemmer.lemmatize(word) for word in train_strings[j]]
            train_strings[j] = ' '.join(train_strings[j])

        # Replace og text data in
        xt[i] = train_strings
        print(str(np.round(i*100/len(str_cols),0)) + '% cleaned')

    t2 = time()
    print('clean time', t2-t1)
    # Return cleaned dataframe of just string columns
    df_text = xt.transpose()
    df_text = pd.DataFrame(df_text,index=df.index, columns= cat_cols)

    return df_text

# Clean up string data
X_text_train = string_cleanup(X_train,cat_cols)
X_text_val   = string_cleanup(X_val,cat_cols)

review_train = X_text_train['review_text'].to_numpy()
review_val   = X_text_val['review_text'].to_numpy()

# Try a neural network - Investigate RNN's more
# encoder = TextVectorization(max_tokens = 10**4)
# encoder.adapt(review_train)
# vocab = encoder.get_vocabulary()
# print('Vocab size: ',len(vocab))
#
# model = Sequential([encoder,
#                    Embedding(input_dim=len(vocab),output_dim=128, mask_zero=True),
#                    Bidirectional(LSTM(128)),
#                    Dense(128, activation='relu'),
#                    Dense(1, activation='softmax')
# ])
#
# model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
#               optimizer='adam', metrics=['accuracy'])
#
# history = model.fit(x=review_train,y=y_train,
#           epochs=10,
#           validation_data=(review_val,y_val),
#           validation_steps=30)
#
# exit()

# ############
# # Trying bert models for nlp preprocessing and classification_report
# # with tensorflow. Not working for some reason?
#
# bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
#
#
# # Initializing BERT layers
# text_input = Input(shape=(), batch_size = 1000, dtype=tf.string, name='text')
# preprocessed_text = bert_preprocess(text_input)
# outputs = bert_encoder(preprocessed_text)
#
#
# # Init nn layers, 2 layers
# l = Dropout(0.8, name="dropout")(outputs['pooled_output'])
# l = Dense(1, activation='softmax', name="classifier")(l)
#
# # Set the model
# model = tf.keras.Model(inputs=text_input, outputs = l)
# #model.summary()
# # model metrics and compilation
# metrics = [tf.keras.metrics.Accuracy(name='accuracy')]#,
#          #  tf.keras.metrics.CategoricalCrossentropy(name='entropy')]
#
# model.compile(optimizer='adam',loss = 'CategoricalCrossentropy', metrics = metrics)
#
#
#
# model.fit(xt[1], y_train, epochs = 10, verbose = 1)
# ############




# Create pipeline and transformer with
# tokenization and frequency weighting for text data
text_preprocessor = Pipeline(steps=[('vect', CountVectorizer(stop_words='english')),
                    ('tfid', TfidfTransformer())])

tuples = [('txt'+str(i),text_preprocessor,cat_cols[i]) for i in range(len(cat_cols))]
text_preprocessor = ColumnTransformer(transformers=tuples)


# Transform text training and validation datasets
X_text_train = text_preprocessor.fit_transform(X_text_train)
X_text_val   = text_preprocessor.transform(X_text_val)

# Dimensionality reduction on sparse matrix from text processing
svd = TruncatedSVD(n_components=5, n_iter=10)
X_text_train = svd.fit_transform(X_text_train)
X_text_val   = svd.transform(X_text_val)


# Convert sparse matrix back into dframe
X_text_train = pd.DataFrame(X_text_train)
X_text_val = pd.DataFrame(X_text_val)
# set indexes
X_text_train = X_text_train.set_index(X_train.index)
X_text_val   = X_text_val.set_index(X_val.index)


# Replace text review columns with num data now
X_train_new = X_train[num_cols].join(X_text_train)
X_val_new   = X_val[num_cols].join(X_text_val)


# Standardize
standard = StandardScaler()
X_train_new = standard.fit_transform(X_text_train)
X_val_new = standard.fit_transform(X_text_val)

#First algorithm, predict based on reviews
model = XGBClassifier(learning_rate=0.01, n_estimators=500, n_jobs=-1)

# Test text predictor
cum = model.fit(X_text_train, y_train)
preds = cum.predict(X_text_val)
print(f1_score(y_val, preds, average='micro'))

# About 35% accurate only


# Next steps to try

#
# # Feature engineering
# # Try feature agglomeration and pca for dimensionality reduction
# shrink = FeatureAgglomeration(n_clusters=5)
# pca = PCA(n_components=5)
#
# # Apply em
# X_train_pca = pca.fit_transform(X_train_new)
# X_val_pca   = pca.transform(X_val_new)
#
# X_train_shrink = shrink.fit_transform(X_train_new)
# X_val_shrink   = shrink.transform(X_val_new)
#
# # Standardize num data
# preprocessor = StandardScaler()
# X_train_new = preprocessor.fit_transform(X_train_new)
# X_val_new   = preprocessor.transform(X_val_new)
#
# X_train_pca = preprocessor.fit_transform(X_train_pca)
# X_val_pca   = preprocessor.transform(X_val_pca)
#
# X_train_shrink = preprocessor.fit_transform(X_train_shrink)
# X_val_shrink   = preprocessor.transform(X_val_shrink)
#
# # Kmeans cluster predictions
# #shrink
#
# plt.figure(1, figsize=(15,15))
# plt.plot(x,y)
#
# y_k     = kmeans.predict(X_val_shrink)
# preds_k.shape
# y_k.shape
# print('Kmeans: f1 score', f1_score(y_true=y_k, y_pred=preds_k, average="micro"),
#       '\nKmeans: confusion_matrix', confusion_matrix(y_k,preds_k))
#
#
# # XGB classifier
# xgb = XGBClassifier(learning_rate=0.01, n_estimators=300, n_jobs=-1)
#
# # Fit to training data. Predict validation data and check accuracy
# xgb.fit(X_train_shrink,y_train)
# preds_tree = xgb.predict(X_val_shrink)
# print('XGB: f1 score', f1_score(y_true=y_val, y_pred=preds_tree, average="micro"),
#       '\nXGB: confusion_matrix', confusion_matrix(y_val,preds_tree))
#
# #XGB: f1 score 0.435605873337618
# #XGB: confusion_matrix [[   0    0    2   16  184  104]
# # [   0    0   21   63  286  175]
# # [   0    0   43  239  927  358]
# # [   0    1   42  553 2907  828]
# # [   0    0   16  405 4713 2086]
# # [   0    0   19  150 2587 3502]]
#
#
#
# # Using neural network
# input_shape = X_train_new.shape[1]
# nn = baseline(input_shape)
# nn.fit(X_train_new, y_train)
# preds_nn = nn.predict(X_val_new)
# print('Neural Net', f1_score(y_true=y_val, y_pred=preds_nn, average="micro"))
#
