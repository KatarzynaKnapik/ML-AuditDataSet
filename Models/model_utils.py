import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix


class ModelUtils:

    @staticmethod
    def draw_label_barchart(df):
        sns.countplot(data=df, x="Risk")
        plt.title("Number of values in Risk column.", fontsize=15);

    @staticmethod
    def split_into_features_and_label(df, label):
        X = df.drop(label, axis=1)
        y = df[label]
        return X, y
    
    @staticmethod
    def divide_into_train_test_datasets(X, y):
        X_train, X_OTHER, y_train, y_OTHER = train_test_split(X, y, test_size=0.2, random_state=1)
        X_eval, X_test, y_eval, y_test = train_test_split(X_OTHER, y_OTHER, test_size=0.5, random_state=1)
        return X, y, X_train, y_train, X_eval, y_eval, X_test,  y_test
    
    @staticmethod
    def randomOverSample(df, X, y):
        oversample = RandomOverSampler(sampling_strategy='minority')
        return oversample.fit_resample(X, y)
    
    @staticmethod
    def fitDefaultModel(X_train, y_train):
        model = RandomForestClassifier(n_estimators=20, max_features="auto")
        model.fit(X_train, y_train)
        return model
    
    @staticmethod
    def printCompleteReport(model, test_X, test_y, pred_y, comment):
        print(f"{comment} Model Report")
        print("\n")
        print(plot_confusion_matrix(model, test_X, test_y))
        print("\n")
        print(classification_report(test_y, pred_y))

    @staticmethod
    def randomForestGridSearch(X_train, y_train, n_estimators: list, max_features: list):
        rfc = RandomForestClassifier()
        param_grid= {  "n_estimators": n_estimators,
                        "max_features": max_features,
                        "criterion": ["gini", "entropy"],
                        "bootstrap": [True, False],
                        "oob_score": [False] 
                    }
        rfc_grid = GridSearchCV(rfc, param_grid)
        rfc_grid.fit(X_train, y_train)
        return rfc_grid
        

       