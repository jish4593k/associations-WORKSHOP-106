import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Function to load data from a file
def load_data():
    file_path = filedialog.askopenfilename(title="Select a file")
    if file_path:
        df = pd.read_table(file_path, sep='\t', header=0)
        return df
    else:
        return None

# Function to perform KMeans clustering
def perform_kmeans(df):
    df_new = df.drop(['origin'], axis=1)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_new)
    df['cluster'] = kmeans.labels_

   
    fig = px.scatter(df, x='acceleration', y='horsepower', color='cluster', title='KMeans Clustering')
    fig.show()

# Function to perform linear regression
def perform_linear_regression(df):
    X = df[['acceleration', 'horsepower']]
    y = df['origin']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Plotly scatter plot for visualizing the regression line
    fig = px.scatter(X_test, x='acceleration', y='horsepower', color=y_test)
    fig.add_trace(px.line(x=X_test['acceleration'], y=model.predict(X_test), mode='lines').data[0])
    fig.update_layout(title='Linear Regression')
    fig.show()

# Function to perform Principal Component Analysis (PCA)
def perform_pca(df):
    df_new = df.drop(['origin'], axis=1)
    pca = PCA(n_components=2)
    coordonnees = pca.fit_transform(df_new)

    # Plotly scatter plot for visualizing PCA
    fig = px.scatter(x=coordonnees[:, 0], y=coordonnees[:, 1], color=df['origin'], title='PCA')
    fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2')
    fig.show()

# Function to perform neural network training using PyTorch
def train_neural_network(df):
    input_dim = len(df.columns) - 1
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df['origin'].values, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(input_dim, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    messagebox.showinfo("Neural Network Training", "Neural network training complete.")

def perform_association_rule_mining(df):
    records = df.astype(str).applymap(lambda x: 'nan' if x != '1' else df.columns[df.columns == x].values[0]).values.tolist()

    association_rules = apriori(records, min_support=0.05, min_confidence=0.3, min_lift=4, min_length=2)
    association_results = list(association_rules)

    # Display association rules using Seaborn
    for item in association_results:
        pair = item[0]
        items = [x for x in pair]

        # Convert association rule data to a DataFrame for easier visualization
        rule_df = pd.DataFrame({
            'Rule': [f"{items[0]} -> {items[1]}"],
            'Support': [item[1]],
            'Confidence': [item[2][0][2]],
            'Lift': [item[2][0][3]]
        })

        # Use Seaborn to display association rules
        sns.barplot(x='Rule', y='Support', data=rule_df)
        plt.title('Association Rule Support')
        plt.show()

        sns.barplot(x='Rule', y='Confidence', data=rule_df)
        plt.title('Association Rule Confidence')
        plt.show()

        sns.barplot(x='Rule', y='Lift', data=rule_df)
        plt.title('Association Rule Lift')
        plt.show()

# Tkinter GUI setup
root = tk.Tk()
root.title("Data Mining Toolkit")

# Load Data button
load_data_button = tk.Button(root, text="Load Data", command=lambda: load_data())
load_data_button.pack()

# KMeans Clustering button
kmeans_button = tk.Button(root, text="Perform KMeans Clustering", command=lambda: perform_kmeans(df))
kmeans_button.pack()

# Linear Regression button
linear_regression_button = tk.Button(root, text="Perform Linear Regression", command=lambda: perform_linear_regression(df))
linear_regression_button.pack()

pca_button = tk.Button(root, text="Perform PCA", command=lambda: perform_pca(df))
pca_button.pack()


nn_button = tk.Button(root, text="Train Neural Network", command=lambda: train_neural_network(df))
nn_button.pack()


association_button = tk.Button(root, text="Perform Association Rule Mining", command=lambda: perform_association_rule_mining(df))
association_button.pack()

exit_button = tk.Button(root, text="Exit", command=root.destroy)
exit_button.pack()

root.mainloop()
