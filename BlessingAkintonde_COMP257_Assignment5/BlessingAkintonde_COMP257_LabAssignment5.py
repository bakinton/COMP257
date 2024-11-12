# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:59:19 2024

@author: Blessing
"""

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


#Load Olivetti faces dataset
data = fetch_olivetti_faces()
X, y = data.data, data.target

#Spliting the dataset into training, validation, and test sets using stratified sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=39)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=39)

#Applying PCA to preserve 99% of the variance
pca = PCA(n_components=0.99, random_state=39)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# Reshaping PCA-transformed data to match the flattened 64x64 structure
X_train_pca_flat = X_train_pca.reshape(X_train_pca.shape[0], -1)
X_val_pca_flat = X_val_pca.reshape(X_val_pca.shape[0], -1)
X_test_pca_flat = X_test_pca.reshape(X_test_pca.shape[0], -1)


# Hyperparameters and configurations
learning_rates = [0.001, 0.0005, 0.0001]
hidden_layer_configs = [[512, 256, 128, 256, 512], [256, 128, 64, 128, 256]]  # Different configurations for hidden layers
kfold = KFold(n_splits=5, shuffle=True, random_state=39)


best_val_loss = float('inf')
best_model = None
best_history = None


# K-fold cross-validation
for lr in learning_rates:
    for hidden_units in hidden_layer_configs:
        fold_num = 1
        for train_idx, val_idx in kfold.split(X_train_pca_flat):
            X_train_fold, X_val_fold = X_train_pca_flat[train_idx], X_train_pca_flat[val_idx]

            # Building the autoencoder model
            input_layer = Input(shape=(X_train_pca_flat.shape[1],))
            x = Dense(hidden_units[0], kernel_regularizer=l2(0.001))(input_layer)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Dropout(0.3)(x)

            for units in hidden_units[1:-1]:
                x = Dense(units, kernel_regularizer=l2(0.001))(x)
                x = BatchNormalization()(x)
                x = LeakyReLU(alpha=0.1)(x)
                x = Dropout(0.3)(x)

            x = Dense(hidden_units[-1], kernel_regularizer=l2(0.001))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)

            output_layer = Dense(X_train_pca_flat.shape[1], activation='sigmoid')(x)

            autoencoder = Model(inputs=input_layer, outputs=output_layer)
            autoencoder.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

            # Training the model
            history = autoencoder.fit(
                X_train_fold, X_train_fold,
                epochs=100,
                batch_size=32,
                validation_data=(X_val_fold, X_val_fold),
                verbose=0
            )

            val_loss = min(history.history['val_loss'])
            print(f"Fold {fold_num} - LR: {lr}, Hidden Units: {hidden_units}, Val Loss: {val_loss:.4f}")
            fold_num += 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = autoencoder
                best_history = history
                

# Ploting the training and validation loss for the best model
plt.plot(best_history.history['loss'], label='Training Loss')
plt.plot(best_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss for Best Model')
plt.show()

# Running the best model with the test set and display original vs reconstructed images
decoded_images = best_model.predict(X_test_pca_flat)

# Displaying original and reconstructed images
n = 10  # Number of images to be displayed
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(pca.inverse_transform(X_test_pca_flat[i]).reshape(64, 64), cmap='gray')
    plt.title("Original image")
    plt.axis('off')

    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(pca.inverse_transform(decoded_images[i]).reshape(64, 64), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()


# Compress data using the encoder model
encoder = Model(inputs=best_model.input, outputs=best_model.layers[len(hidden_layer_configs[0])].output)
X_train_encoded = encoder.predict(X_train_pca_flat)
X_test_encoded = encoder.predict(X_test_pca_flat)

#Training a base model for comparison
# Logistic Regression on original and encoded data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train_pca_flat, y_train)
y_pred_original = base_model.predict(X_test_pca_flat)
accuracy_original = accuracy_score(y_test, y_pred_original)

print(f"Base Model Accuracy on Original Data: {accuracy_original:.4f}")


# Training logistic regression on encoded (compressed) data
base_model_encoded = LogisticRegression(max_iter=1000)
base_model_encoded.fit(X_train_encoded, y_train)
y_pred_encoded = base_model_encoded.predict(X_test_encoded)
accuracy_encoded = accuracy_score(y_test, y_pred_encoded)

print(f"Base Model Accuracy on Encoded Data: {accuracy_encoded:.4f}")