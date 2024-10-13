# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 01:29:18 2024

@author: Blessing
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Olivetti faces dataset
data = fetch_olivetti_faces()
X, y = data.data, data.target

# Split the dataset into training, validation, and test sets using stratified sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=39)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=39)

# Standardize the data (SVM is sensitive to feature scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the SVM classifier
svm_clf = SVC(kernel='linear', random_state=39)

# Define K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=39)

# Train and evaluate using cross-validation
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=kf)

# Print the cross-validation results
print(f"K-Fold Cross Validation Scores (SVM): {svm_scores}")
print(f"Mean SVM Score: {svm_scores.mean()}")

# Training on the full training set and evaluate on the validation set
svm_clf.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = svm_clf.predict(X_val)

# Evaluate the predictions on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")

# Display classification report to evaluate the performance on the validation set
print(classification_report(y_val, y_val_pred, zero_division=0))


from sklearn.decomposition import PCA

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50, random_state=39)  
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# Perform Agglomerative Hierarchical Clustering (AHC) using Euclidean Distance

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Perform Agglomerative Clustering with Euclidean Distance
ahc_euclidean = AgglomerativeClustering(n_clusters=40, linkage='ward')
labels_euclidean = ahc_euclidean.fit_predict(X_train_pca)

# Calculate silhouette score
sil_score_euclidean = silhouette_score(X_train_pca, labels_euclidean)
print(f"Silhouette Score (Euclidean): {sil_score_euclidean}")

# Perform Agglomerative Hierarchical Clustering (AHC) using Minkowski Distance

from sklearn.metrics import pairwise_distances

# Compute Minkowski Distance with p=3
distance_matrix_minkowski = pairwise_distances(X_train_pca, metric='minkowski', p=3)

# Perform Agglomerative Clustering
ahc_minkowski = AgglomerativeClustering(n_clusters=40, metric='precomputed', linkage='complete')
labels_minkowski = ahc_minkowski.fit_predict(distance_matrix_minkowski)

# Calculate silhouette score
sil_score_minkowski = silhouette_score(X_train_pca, labels_minkowski)
print(f"Silhouette Score (Minkowski, p=3): {sil_score_minkowski}")


# Perform Agglomerative Hierarchical Clustering (AHC) using cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity

# Compute Cosine Similarity distance matrix
distance_matrix_cosine = 1 - cosine_similarity(X_train_pca)

# Perform Agglomerative Clustering with the precomputed distance matrix
ahc_cosine = AgglomerativeClustering(n_clusters=40, metric='precomputed', linkage='complete')
labels_cosine = ahc_cosine.fit_predict(distance_matrix_cosine)

# Calculate silhouette score
sil_score_cosine = silhouette_score(X_train_pca, labels_cosine, metric='cosine')
print(f"Silhouette Score (Cosine Similarity): {sil_score_cosine}")

#Graphical Representation
import numpy as np
import matplotlib.pyplot as plt

# Silhouette scores for different similarity measures
similarity_measures = ['Euclidean', 'Minkowski (p=3)', 'Cosine Similarity']
silhouette_scores = [ 0.23445415496826172, 0.1960592120885849, 0.32095515727996826]

# Plotting the silhouette scores
plt.figure(figsize=(8, 6))
plt.bar(similarity_measures, silhouette_scores, color=['pink', 'purple', 'green'])
plt.xlabel('Similarity Measures')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Similarity Measures')
plt.ylim(0, 0.4)

# Adding value labels on top of the bars
for i, score in enumerate(silhouette_scores):
    plt.text(i, score + 0.01, f'{score:.3f}', ha='center', fontsize=12)

plt.show()

#Using the silhouette score approach to choose the number of clusters for 4(a), 4(b), and 4(c)

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

# Function to perform Agglomerative Clustering and return silhouette score for a range of clusters
def calculate_silhouette_scores(X, metric, cluster_range, p=None):
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        if metric == 'minkowski' and p is not None:
            distance_matrix = pairwise_distances(X, metric='minkowski', p=p)
            ahc = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
            labels = ahc.fit_predict(distance_matrix)
        elif metric == 'cosine':
            distance_matrix = 1 - cosine_similarity(X)
            ahc = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
            labels = ahc.fit_predict(distance_matrix)
        else:
            ahc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
            labels = ahc.fit_predict(X)
        
        score = silhouette_score(X, labels)
        silhouette_scores.append((n_clusters, score))
    
    return silhouette_scores

# Define the range of clusters to test
cluster_range = range(10, 201, 10)  

# Calculate silhouette scores for Euclidean Distance
silhouette_scores_euclidean = calculate_silhouette_scores(X_train_pca, 'euclidean', cluster_range)

# Calculate silhouette scores for Minkowski Distance with p=3
silhouette_scores_minkowski = calculate_silhouette_scores(X_train_pca, 'minkowski', cluster_range, p=3)

# Calculate silhouette scores for Cosine Similarity
silhouette_scores_cosine = calculate_silhouette_scores(X_train_pca, 'cosine', cluster_range)

# Find the number of clusters with the best silhouette score
best_euclidean = max(silhouette_scores_euclidean, key=lambda x: x[1])
best_minkowski = max(silhouette_scores_minkowski, key=lambda x: x[1])
best_cosine = max(silhouette_scores_cosine, key=lambda x: x[1])

print(f"Best Silhouette Score for Euclidean: {best_euclidean[1]} with {best_euclidean[0]} clusters")
print(f"Best Silhouette Score for Minkowski: {best_minkowski[1]} with {best_minkowski[0]} clusters")
print(f"Best Silhouette Score for Cosine: {best_cosine[1]} with {best_cosine[0]} clusters")

#Silhouette diagrams 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

# Function to plot the silhouette diagram for a given clustering result
def plot_silhouette(X, cluster_labels, n_clusters, metric):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Computing the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label silhouette plots with cluster number at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute new y_lower for the next plot
        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for various clusters.")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # The 2nd plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    ax2.set_title(f"Clustered data visualization ({metric} metric)")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(f"Silhouette analysis for {metric.capitalize()} clustering with n_clusters = {n_clusters}",
                 fontsize=14, fontweight="bold")

    plt.show()

# Function to perform Agglomerative Clustering and plot silhouette for each metric
def perform_and_plot_silhouette(X, n_clusters, metric, p=None):
    if metric == 'minkowski' and p is not None:
        distance_matrix = pairwise_distances(X, metric='minkowski', p=p)
        ahc = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
        labels = ahc.fit_predict(distance_matrix)
    elif metric == 'cosine':
        distance_matrix = 1 - cosine_similarity(X)
        ahc = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
        labels = ahc.fit_predict(distance_matrix)
    else:
        ahc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        labels = ahc.fit_predict(X)

    plot_silhouette(X, labels, n_clusters, metric)

# Example for Euclidean, Minkowski, and Cosine with best clusters obtained previously
best_n_clusters_euclidean = best_euclidean[0]
best_n_clusters_minkowski = best_minkowski[0]
best_n_clusters_cosine = best_cosine[0]

# Plot silhouette analysis for Euclidean
perform_and_plot_silhouette(X_train_pca, best_n_clusters_euclidean, 'euclidean')

# Plot silhouette analysis for Minkowski (p=3)
perform_and_plot_silhouette(X_train_pca, best_n_clusters_minkowski, 'minkowski', p=3)

# Plot silhouette analysis for Cosine
perform_and_plot_silhouette(X_train_pca, best_n_clusters_cosine, 'cosine')

#Graphical Representation
import matplotlib.pyplot as plt

# Extract cluster numbers and scores for each measure
clusters_euclidean, scores_euclidean = zip(*silhouette_scores_euclidean)
clusters_minkowski, scores_minkowski = zip(*silhouette_scores_minkowski)
clusters_cosine, scores_cosine = zip(*silhouette_scores_cosine)

# Plotting the silhouette scores for each measure
plt.figure(figsize=(10, 6))
plt.plot(clusters_euclidean, scores_euclidean, label='Euclidean', marker='o')
plt.plot(clusters_minkowski, scores_minkowski, label='Minkowski (p=3)', marker='s')
plt.plot(clusters_cosine, scores_cosine, label='Cosine Similarity', marker='^')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters for Different Distance Metrics')
plt.legend()
plt.grid(True)
plt.show()

# Using the set from (4(a), 4(b), or 4(c)) to train a classifier as in (3) using k-fold cross validation

from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Applying PCA to both training and validation sets
pca = PCA(n_components=50, random_state=39)  
X_train_pca = pca.fit_transform(X_train)  
X_val_pca = pca.transform(X_val)         

# Using pseudo-labels generated from clustering
best_labels = labels_euclidean  

# Train a new SVM classifier on the PCA-reduced dataset using pseudo-labels
svm_clf = SVC(kernel='linear', random_state=39)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=39)

# Perform K-Fold cross-validation with pseudo-labels
svm_scores = cross_val_score(svm_clf, X_train_pca, best_labels, cv=kf)

# Print the cross-validation results
print(f"K-Fold Cross Validation Scores (SVM with Euclidean Clusters): {svm_scores}")
print(f"Mean SVM Score: {svm_scores.mean()}")

# Train on the full training set using the PCA-transformed data and pseudo-labels
svm_clf.fit(X_train_pca, best_labels)

# Make predictions on the PCA-transformed validation set
y_val_pred = svm_clf.predict(X_val_pca)


print(f"Validation Accuracy (using pseudo-labels): {accuracy_score(best_labels[:len(y_val_pred)], y_val_pred)}")

# Display the classification report for pseudo-labels prediction
print("Classification Report (Clusters as Labels):")
print(classification_report(best_labels[:len(y_val_pred)], y_val_pred, zero_division=0))

# Display the confusion matrix for further evaluation
print("Confusion Matrix (Clusters as Labels):")
print(confusion_matrix(best_labels[:len(y_val_pred)], y_val_pred))

# Train on original labels
svm_clf.fit(X_train_pca, y_train)

# Make predictions on the validation set
y_val_pred = svm_clf.predict(X_val_pca)

# Evaluate the predictions on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy (Original Labels): {val_accuracy}")
print(classification_report(y_val, y_val_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))



