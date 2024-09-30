# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:22:57 2024

@author: Blessing
"""

import numpy as np


from sklearn.datasets import fetch_olivetti_faces

# Loading the Olivetti faces dataset
Blessing_Faces = fetch_olivetti_faces(shuffle=True, random_state=39)

X = Blessing_Faces.data 
 
y = Blessing_Faces.target

from sklearn.model_selection import train_test_split

# Spliting the dataset into training will be (70%) and will be temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=39)

# Spliting the temp dataset (15%) will be used for validation and (15%) for test 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=39)

print(f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}, Test set size: {X_test.shape}")

#Train a classifier Using k-Fold Cross Valiadtion

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

# Using RandomForest for classification
clf = RandomForestClassifier(n_estimators=100, random_state=39)

# k-fold cross-validation 
scores = cross_val_score(clf, X_train, y_train, cv=5)

print(f"Cross-validation scores: {scores}")

print(f"Mean CV accuracy: {scores.mean():.2f}")

#Using K-Means for Dimensionality Reduction

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

# Number of clusters using the silhouette score
best_k = 0
best_score = -1
for k in range(2, 21):  
    kmeans = KMeans(n_clusters=k, random_state=39)
    kmeans.fit(X_train)
    score = silhouette_score(X_train, kmeans.labels_)
    if score > best_score:
        best_k = k
        best_score = score

print(f"Optimal number of clusters based on silhouette score: {best_k}")

# Use step (4) to train a classifier 

# Applying K-Means with the number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=39)

# Fit KMeans to the training data
kmeans.fit(X_train)

X_train_kmeans = kmeans.transform(X_train)

X_val_kmeans = kmeans.transform(X_val)

# Training the classifier using the reduced dataset
clf_kmeans = RandomForestClassifier(n_estimators=100, random_state=39)

clf_kmeans.fit(X_train_kmeans, y_train)

# Evaluate on the validation set
val_accuracy_kmeans = clf_kmeans.score(X_val_kmeans, y_val)

print(f"Validation accuracy after dimensionality reduction with K-Means: {val_accuracy_kmeans * 100:.2f}%")

# Adjusting
# To find the best number of clusters using silhouette score

def tune_kmeans(X_train, X_val, y_val, k_values):
    best_k = None
    best_score = -1
    best_val_accuracy = 0

k_values = [10, 20, 30, 40, 50]

for k in k_values:
    print(f"Training K-Means with {k} clusters...")

# Applying K-Means with current k
        
kmeans = KMeans(n_clusters=k, random_state=39)
        
kmeans.fit(X_train)

# Transform the data
X_train_kmeans = kmeans.transform(X_train)

X_val_kmeans = kmeans.transform(X_val)

# Calculating silhouette score for current k
silhouette_avg = silhouette_score(X_train, kmeans.labels_)

print(f"Silhouette Score for k={k}: {silhouette_avg:.4f}")

classifier = RandomForestClassifier(n_estimators=100, random_state=39)

# Train the classifier with the reduced data (after K-Means)
classifier.fit(X_train_kmeans, y_train)

# Evaluate the classifier on the validation set
val_accuracy_kmeans = classifier.score(X_val_kmeans, y_val)

print(f"Validation Accuracy for k={k}: {val_accuracy_kmeans * 100:.2f}%\n")

# Update if there is a better silhouette score and accuracy
if silhouette_avg > best_score: 
    best_score = silhouette_avg
    best_k = k
    best_val_accuracy = val_accuracy_kmeans

    print(f"Best number of clusters: {best_k} with silhouette score: {best_score:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    
# List of k values to try
k_values = [10, 20, 30, 40, 50]

# Tuning K-Means and finding the best number of clusters
best_k = tune_kmeans(X_train, X_val, y_val, k_values)

# Output the best k value
print(f"Best overall number of clusters: {best_k} with validation accuracy: {best_val_accuracy * 100:.2f}%")

#Question6
#Applying Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Loading the Olivetti Faces dataset
Blessing_Faces = fetch_olivetti_faces(shuffle=True, random_state=39)
X = Blessing_Faces.data  
y = Blessing_Faces.target  

#Preprocess the images using StandardScaler (normalizing the pixel intensity values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Applying t-SNE for non-linear dimensionality reduction
tsne = TSNE(n_components=2, random_state=39)
X_tsne = tsne.fit_transform(X_scaled)

# Plotting t-SNE reduced data to visualize clusters
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title('t-SNE Visualization of Olivetti Faces Dataset')
plt.show()

# Apply DBSCAN on t-SNE reduced data with different parameters
eps_values = [1.5, 2.0, 2.5]  
min_samples_values = [3, 5]  
distance_metrics = ['cosine', 'manhattan', 'euclidean']


for eps in eps_values:
    for min_samples in min_samples_values:
        for metric in distance_metrics:
            print(f"\nTesting DBSCAN with eps={eps}, min_samples={min_samples}, metric={metric}")


#Apply DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            dbscan_labels = dbscan.fit_predict(X_tsne)

# Number of clusters and noise points
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise = list(dbscan_labels).count(-1)
            
            if n_clusters > 1:
                silhouette_avg = silhouette_score(X_tsne, dbscan_labels)
                print(f"Estimated number of clusters: {n_clusters}")
                print(f"Number of noise points: {n_noise}")
                print(f"Silhouette Score: {silhouette_avg:.4f}")
            else:
                print(f"Estimated number of clusters: {n_clusters}")
                print(f"Number of noise points: {n_noise}")
                print("Too few clusters to calculate silhouette score.")


#Visualization After applying DBSCAN

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

#DBSCAN to the t-SNE-reduced dataset
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
dbscan_labels = dbscan.fit_predict(X_tsne)

# Updated Plotting function to visualize clusters
def plot_clusters(X, labels, title):
    # Unique cluster labels
    unique_labels = set(labels)
    
    # Create a color map for clusters
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize=(8, 6))  # Set figure size
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black is used for noise
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = X[class_member_mask]

        plt.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolor='k', s=50)  

    plt.title(title)
    plt.show()

plot_clusters(X_tsne, dbscan_labels, f"DBSCAN: eps={eps}, min_samples={min_samples}, metric={metric}")

#Using make_blobs for better visualization

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Generate synthetic data using make_blobs
X_blobs, y_blobs = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=39)

# Standardize the data
scaler = StandardScaler()
X_scaled_blobs = scaler.fit_transform(X_blobs)


# Apply DBSCAN on the synthetic data
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
dbscan_labels_blobs = dbscan.fit_predict(X_scaled_blobs)

# Plotting function to visualize clusters
def plot_clusters(X, labels, title):
    unique_labels = set(labels)
    
    # Create a color map for clusters
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize=(8, 6))  
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise
            col = [0, 0, 0, 1]  

        class_member_mask = (labels == k)
        xy = X[class_member_mask]

        plt.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolor='k', s=50, label=f"Cluster {k}" if k != -1 else "Noise")

    plt.title(title)
    plt.legend()  # A legend to differentiate clusters
    plt.show()

# Visualize clusters using plot_clusters
plot_clusters(X_scaled_blobs, dbscan_labels_blobs, "Discovering Groupings with DBSCAN on Scaled Blob Data")



