# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 22:40:30 2024

@author: Blessing
"""

from sklearn.datasets import fetch_openml 
import matplotlib.pyplot as plt

#retrieving and loading the mnist_784 dataset 
mnist_Blessing = fetch_openml('mnist_784', version=1)

# Getting the data 
X_Blessing, y_Blessing = mnist_Blessing["data"], mnist_Blessing["target"]
print(f"Loaded dataset with {X_Blessing.shape[0]} instances and {X_Blessing.shape[1]} features.")

# Reshaping the data 
X_reshaped_Blessing = X_Blessing.values.reshape(-1, 28, 28)  

# Looping through the dataset and display each digit
for i in range(100):  
    plt.figure(figsize=(1, 1))  
    plt.imshow(X_reshaped_Blessing[i], cmap="gray") 
    plt.title(f"Digit: {y_Blessing[i]}") 
    plt.axis('off')  
    plt.show()
    

# Using PCA to Retrieve the 1st and 2nd Principal Component and Output Their Explained Variance Ratio
from sklearn.decomposition import PCA
  
# Perform PCA to reduce the data to 2 components
pca_Blessing = PCA(n_components=2)
X_pca_Blessing = pca_Blessing.fit_transform(X_Blessing)


# Output explained variance ratio
explained_variance_Blessing = pca_Blessing.explained_variance_ratio_
print(f"Explained variance by the 1st component: {explained_variance_Blessing[0]:.4f}")
print(f"Explained variance by the 2nd component: {explained_variance_Blessing[1]:.4f}")

# Plotting the Projections of the 1st and 2nd Principal Component onto a 1D Hyperplane
plt.scatter(X_pca_Blessing[:, 0], X_pca_Blessing[:, 1], c=y_Blessing.astype(int), cmap="inferno", alpha=0.5)
plt.colorbar(label="Digit Label")
plt.title("Projections of the 1st and 2nd Principal Components")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()


# Using incremental PCA to reduce the dimensionality to 154 dimensions
from sklearn.decomposition import IncrementalPCA

# Reducing the dimensionality to 154 dimensions using Incremental PCA
ipca_Blessing = IncrementalPCA(n_components=154, batch_size=200)
X_ipca_Blessing = ipca_Blessing.fit_transform(X_Blessing)
print(f"The reduced data shape: {X_ipca_Blessing.shape}")

# Reconstruct the digits from reduced dimensions
X_reconstructed_Blessing = ipca_Blessing.inverse_transform(X_ipca_Blessing)

# Display original and compressed digits side by side
for i in range(5):  # Display the first 5 digits as examples
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(X_reshaped_Blessing[i], cmap='gray')
    ax[0].set_title(f"Original Digit {y_Blessing[i]}")
    
    ax[1].imshow(X_reconstructed_Blessing[i].reshape(28, 28), cmap='gray')
    ax[1].set_title(f"Compressed Digit {y_Blessing[i]}")
    
    plt.show()

#Section 2
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate Swiss roll dataset 
X_Blessing, t_Blessing = make_swiss_roll(n_samples=3000, noise=0.0, random_state=39)

# Plot the 3D Swiss roll
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot 
ax.scatter(X_Blessing[:, 0], X_Blessing[:, 1], X_Blessing[:, 2], c=t_Blessing, cmap=plt.get_cmap('Spectral'), s=10, alpha=0.8)
plt.title("Swiss Roll Dataset")
ax.view_init(azim=-70, elev=20)  # Adjust the viewing angles to see the twists
plt.show()


from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

# Scaling the data
scaler_Blessing = StandardScaler()
X_Blessing_scaled = scaler_Blessing.fit_transform(X_Blessing)

# Kernel PCA with Linear Kernel
kpca_Blessing_linear = KernelPCA(n_components=2, kernel='linear')
X_Blessing_kpca_linear = kpca_Blessing_linear.fit_transform(X_Blessing_scaled)

# Kernel PCA with RBF Kernel
kpca_Blessing_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_Blessing_kpca_rbf = kpca_Blessing_rbf.fit_transform(X_Blessing_scaled)

# Kernel PCA with Sigmoid Kernel
kpca_Blessing_sigmoid = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.01)
X_Blessing_kpca_sigmoid = kpca_Blessing_sigmoid.fit_transform(X_Blessing_scaled)


# Plotting the kPCA with the results
def plot_kpca_Blessing(X_Blessing_kpca, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_Blessing_kpca[:, 0], X_Blessing_kpca[:, 1], c=t_Blessing, cmap='Spectral', s=10)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Plotting kPCA results for the three kernels
plot_kpca_Blessing(X_Blessing_kpca_linear, "kPCA with Linear Kernel")
plot_kpca_Blessing(X_Blessing_kpca_rbf, "kPCA with RBF Kernel")
plot_kpca_Blessing(X_Blessing_kpca_sigmoid, "kPCA with Sigmoid Kernel")


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import KernelPCA

# Converting continuous target values into categorical bins
binner_Blessing = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
t_Blessing_binned = binner_Blessing.fit_transform(t_Blessing.reshape(-1, 1)).ravel()

# Spliting the data
X_Blessing_train, X_Blessing_test, t_Blessing_train, t_Blessing_test = train_test_split(
    X_Blessing_scaled, t_Blessing_binned, test_size=0.2, random_state=39
)


# Building the pipeline with kPCA and Logistic Regression
pipeline_Blessing = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])


# Defining the parameter grid
param_grid_Blessing = [
    {
        "kpca__kernel": ["rbf", "sigmoid"],
        "kpca__gamma": [0.01, 0.03, 0.05, 0.1, 0.5]
    }
]

# Using the GridSearchCV to find the best parameters
grid_search_Blessing = GridSearchCV(pipeline_Blessing, param_grid_Blessing, cv=3)
grid_search_Blessing.fit(X_Blessing_train, t_Blessing_train)

# Print out the best parameters
print(f"Best Parameters for the model: {grid_search_Blessing.best_params_}")

# Getting the best kPCA model from GridSearchCV
best_kpca_Blessing = grid_search_Blessing.best_estimator_.named_steps["kpca"]
X_Blessing_kpca_best = best_kpca_Blessing.transform(X_Blessing_test)

# Plotting the results using the best kPCA model
def plot_kpca_Blessing(X_Blessing_kpca, t_labels_Blessing, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_Blessing_kpca[:, 0], X_Blessing_kpca[:, 1], c=t_labels_Blessing, cmap='Spectral', s=10)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Class Label')
    plt.show()

plot_kpca_Blessing(X_Blessing_kpca_best, t_Blessing_test, "The Model's Best kPCA Results with Logistic Regression")
