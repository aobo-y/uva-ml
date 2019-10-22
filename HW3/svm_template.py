# Starting code for UVA CS 4501 ML- SVM

import numpy as np
np.random.seed(37)
import random

from sklearn.svm import SVC
# Att: You're not allowed to use modules other than SVC in sklearn, i.e., model_selection.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas

# Dataset information
# the column names (names of the features) in the data files
# you can use this information to preprocess the features
col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country']
col_names_y = ['label']

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                  'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'native-country']

# 1. Data loading from file and pre-processing.
# Hint: Feel free to use some existing libraries for easier data pre-processing.
# For example, as a start you can use one hot encoding for the categorical variables and normalization
# for the continuous variables.

# make them global to save preprocessor fitted by training
le, ohe, ss = None, None, None

def load_data(csv_file_path):
    global le, ohe, ss

    # your code here
    df = pandas.read_csv(csv_file_path, sep=', ', names=col_names_x+col_names_y, engine='python')

    if not le:
        le = LabelEncoder()
        le.fit(['<=50K', '>50K'])
    y = le.transform(df[col_names_y[0]].values)

    # rm feature native-country
    filtered_cat_cols = categorical_cols[:-1]

    x_cat = df[filtered_cat_cols].values
    if not ohe:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        ohe.fit(x_cat)
    x_cat = ohe.transform(x_cat)

    x_num = df[numerical_cols].values
    if not ss:
        ss = StandardScaler()
        ss.fit(x_num)
    x_num = ss.transform(x_num)

    x = np.concatenate([x_num, x_cat], axis=1)

    return x, y


# 2. Select best hyperparameter with cross validation and train model.
# Attention: Write your own hyper-parameter candidates.

def fold(x, y, i, nfolds):
    # your code
    fold_size = int(len(x) / nfolds)

    x_train = np.concatenate([
        x[:i * fold_size],
        x[(i + 1) * fold_size:]
    ])
    y_train = np.concatenate([
        y[:i * fold_size],
        y[(i + 1) * fold_size:]
    ])

    x_test = x[i * fold_size:(i + 1) * fold_size]
    y_test = y[i * fold_size:(i + 1) * fold_size]

    return x_train, y_train, x_test, y_test

def train_and_select_model(training_csv):
    # load data and preprocess from filename training_csv
    x_train, y_train = load_data(training_csv)
    # hard code hyperparameter configurations, an example:
    param_set = [
        {'kernel': 'linear', 'C': 1},
        {'kernel': 'linear', 'C': 10},
        # {'kernel': 'linear', 'C': 100},
        # {'kernel': 'linear', 'C': 1000},

        # {'kernel': 'rbf', 'C': .1},
        {'kernel': 'rbf', 'C': 1},
        {'kernel': 'rbf', 'C': 10},
        {'kernel': 'rbf', 'C': 100},
        # {'kernel': 'rbf', 'C': 1000},

        {'kernel': 'poly', 'C': 1, 'degree': 1},
        {'kernel': 'poly', 'C': 1, 'degree': 3},
        {'kernel': 'poly', 'C': 1, 'degree': 5},
        # {'kernel': 'poly', 'C': 1, 'degree': 7},
    ]
    # your code here
    # iterate over all hyperparameter configurations
    # perform 3 FOLD cross validation
    # print cv scores for every hyperparameter and include in pdf report
    # select best hyperparameter from cv scores, retrain model
    nfolds = 3
    best_score = 0
    best_param = None

    for param in param_set:
        trn_fold_accuracy = []
        val_fold_accuracy = []
        for i in range(nfolds):
            x_trn, y_trn, x_val, y_val = fold(x_train, y_train, i, nfolds)

            model = SVC(**param, gamma='auto')
            model.fit(x_trn, y_trn)
            trn_fold_accuracy.append(model.score(x_trn, y_trn))
            val_fold_accuracy.append(model.score(x_val, y_val))

        trn_accuracy = sum(trn_fold_accuracy) / nfolds
        val_accuracy = sum(val_fold_accuracy) / nfolds
        print(param, 'trn accuracy: %.4f' % trn_accuracy, 'val accuracy: %.4f' % val_accuracy)

        if val_accuracy > best_score:
            best_score = val_accuracy
            best_param = param

    print('best param:', best_param)
    best_model = SVC(**best_param, gamma='auto')
    best_model.fit(x_train, y_train)

    return best_model, best_score

# predict for data in filename test_csv using trained model
def predict(test_csv, trained_model):
    x_test, _ = load_data(test_csv)
    predictions = trained_model.predict(x_test)
    return predictions

# save predictions on test data in desired format
def output_results(predictions):
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            if pred == 0:
                f.write('<=50K\n')
            else:
                f.write('>50K\n')

if __name__ == '__main__':
    training_csv = 'salary.labeled.csv'
    testing_csv = 'salary.2Predict.csv'
    # fill in train_and_select_model(training_csv) to
    # return a trained model with best hyperparameter from 3-FOLD
    # cross validation to select hyperparameters as well as cross validation score for best hyperparameter.
    # hardcode hyperparameter configurations as part of train_and_select_model(training_csv)
    trained_model, cv_score = train_and_select_model(training_csv)

    print('The best model was scored %.2f' % cv_score)
    # use trained SVC model to generate predictions
    predictions = predict(testing_csv, trained_model)
    # Don't archive the files or change the file names for the automated grading.
    # Do not shuffle the test dataset
    output_results(predictions)
    # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
