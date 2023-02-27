import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    
    @staticmethod
    def draw_corr_heatmap(df):
        plt.figure(figsize=(20, 12))
        sns.heatmap(df.corr(), annot=True)
        plt.title("Heatmap - correlation between all predictors.", fontsize=13)
        plt.show()

    @staticmethod
    def calculate_corr(df, column1, column2="Risk"):
        return df[column1].corr(df[column2])
    
    @staticmethod
    def detect_highly_correlated_columns_to_label(df, corr_threshold):
        highly_correlated_columns = {}

        for column in df.columns:
            correlation = EDA.calculate_corr(df, column)
            if correlation > corr_threshold or correlation < -corr_threshold:
                highly_correlated_columns[column] = correlation
        return highly_correlated_columns
    
    @staticmethod
    def count_percentile(data, percentile, interpolation = 'midpoint'):
        return np.percentile(data, percentile)
    
    @staticmethod
    def detect_outlier_iqr(column):
        outliers = []
        sorted_col =  np.sort(column)
        
        Q1 = EDA.count_percentile(sorted_col, 25)  
        Q3 = EDA.count_percentile(sorted_col, 75) 
        IQR = Q3 - Q1 
        
        low_limit = Q1 - 2 * IQR
        up_limit = Q3 + 2 * IQR
        print("low_limit:", low_limit, "up_limit:", up_limit)
        
        for i, value in enumerate(sorted_col):
            if ((value < low_limit) or (value > up_limit)):
                print("outlier value:", value)
                outliers.append(i)
        
        return outliers
    
    @staticmethod
    def detect_outliers_boxplot(df, column, treshold):
        mask = df[column] > treshold
        rows = df.loc[mask]
        print(pd.DataFrame(rows))
        print("\n")
        print(f'Potential number of rows to drop: {len(df.loc[mask])}')   
        print("\n")
        print(f"Indexes to drop: {rows.index}")