import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tkinter import messagebox
from utils import read_csv, write_csv

def label_encoding():
    df = pd.read_csv('Sentiment_dataset/sentimentdataset.csv')
    text_columns = df.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for column in text_columns:
        df[column] = encoder.fit_transform(df[column])
    output_file_path = 'Encoded_dataset/encoded_dataset.csv'
    df.to_csv(output_file_path, index=False)
    messagebox.showinfo("Success", "Label encoding applied and CSV file generated.")

def generate_heatmap():
    df = pd.read_csv('Encoded_dataset/encoded_dataset.csv')
    corr = df.corr(numeric_only=True)
    relevant_features = corr['Text'][abs(corr['Text']) > 0.1].index.tolist()
    df[relevant_features].to_csv('Heatmap/relevant_dataset.csv', index=False)
    sns.heatmap(corr, annot=True, fmt="0.02f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig('Heatmap/correlation_heatmap.pdf')
    plt.show()
    messagebox.showinfo("Success", "Heatmap and CSV generated.")

def normalization():
    data = read_csv('Heatmap/relevant_dataset.csv')
    # Add normalization code here
    # Save normalized data
    write_csv('Normalization/normalized_dataset.csv', data)
    messagebox.showinfo("Success", "Normalization applied and CSV file generated.")

def apply_sigmoid():
    df = pd.read_csv('Normalization/normalized_dataset.csv')
    df['Sentiment_Sigmoid'] = 1 / (1 + np.exp(-df['Sentiment']))
    df['Sentiment_Binary'] = (df['Sentiment_Sigmoid'] >= 0.66).astype(int)
    df.to_csv('Sigmoid_output/sigmoid_output.csv', index=False)
    messagebox.showinfo("Success", "Sigmoid applied and CSV file generated.")
