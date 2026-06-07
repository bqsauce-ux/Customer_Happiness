import pandas as pd

df = pd.read_csv('data/raw/ACME-HappinessSurvey2020.csv')

def replace_outliers_with_median(df, column):
    """
    Replace outliers in a column using the IQR method.
    Outliers are replaced with the column median.
    """

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    median = df[column].median()

    df.loc[df[column] < lower_bound, column] = median
    df.loc[df[column] > upper_bound, column] = median

    return df

survey_columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']

for col in survey_columns:
    df = replace_outliers_with_median(df, col)
 
df.to_csv('data/processed/ACME-HappinessSurvey2020.csv', index=False)
