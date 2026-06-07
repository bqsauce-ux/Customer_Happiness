{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e9696e5c-0368-40b6-ba1f-37aa375b5fcd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "def replace_outliers_with_median(df, column):\n",
    "    \"\"\"\n",
    "    Replace outliers in a column using the IQR method.\n",
    "    Outliers are replaced with the column median.\n",
    "    \"\"\"\n",
    "\n",
    "    Q1 = df[column].quantile(0.25)\n",
    "    Q3 = df[column].quantile(0.75)\n",
    "\n",
    "    IQR = Q3 - Q1\n",
    "\n",
    "    lower_bound = Q1 - 1.5 * IQR\n",
    "    upper_bound = Q3 + 1.5 * IQR\n",
    "\n",
    "    median = df[column].median()\n",
    "\n",
    "    df.loc[df[column] < lower_bound, column] = median\n",
    "    df.loc[df[column] > upper_bound, column] = median\n",
    "\n",
    "    return df\n",
    "\n",
    "\n",
    "survey_columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']\n",
    "\n",
    "for col in survey_columns:\n",
    "    df = replace_outliers_with_median(df, col)\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
