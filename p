# Import necessary libraries
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Step 2: Apply PCA (reduce from 4D → 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 3: Print explained variance
print("Explained variance ratio (how much info each component holds):")
print(pca.explained_variance_ratio_)

# Step 4: Visualize the PCA result
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset (2D Visualization)')
plt.colorbar(label='Class Label')
plt.show()
