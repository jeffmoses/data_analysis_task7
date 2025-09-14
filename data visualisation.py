# --- Task 1: Load and Explore the Dataset ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset, which is a classic dataset for data analysis.
# It contains data on three types of iris flowers.
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add the species column to the DataFrame
df['species'] = pd.Series(iris.target_names)[iris.target]

print("--- Task 1: Dataset Exploration ---")

# Display the first few rows to inspect the data's structure.
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Get information about the dataset, including data types and non-null counts.
print("\nDataset Information:")
df.info()

# Cleaning the dataset:
# The Iris dataset is already very clean, with no missing values.
# However, this is an example of how you would handle missing data
# in a real-world scenario.
print("\nChecking for missing values:")
print(df.isnull().sum())

# Example of how you would drop rows with missing values:
# df_cleaned = df.dropna()

# Example of how you would fill missing values with the mean of the column:
# df_filled = df.fillna(df.mean(numeric_only=True))


# --- Task 2: Basic Data Analysis ---

print("\n--- Task 2: Basic Data Analysis ---")

# Compute basic statistics for numerical columns.
# The describe() method provides count, mean, std, min, max, and quartiles.
print("\nBasic statistics of the numerical columns:")
print(df.describe())

# Group the data by 'species' and compute the mean of 'sepal length (cm)' for each group.
print("\nMean sepal length by species:")
species_mean = df.groupby('species')['sepal length (cm)'].mean()
print(species_mean)

# Identify patterns:
# Based on the output, we can observe that different species of iris flowers
# have distinct average sepal lengths. For example, 'setosa' has a smaller
# average sepal length compared to 'virginica' and 'versicolor'.


# --- Task 3: Data Visualization ---

print("\n--- Task 3: Data Visualization ---")

# Set up the plot area with a single figure and a grid of subplots (2 rows, 2 columns).
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Line Chart
# We'll plot sepal length over the index of the DataFrame to simulate a trend.
# The Iris dataset does not have a time component, so this is for demonstration.
axes[0, 0].plot(df.index, df['sepal length (cm)'])
axes[0, 0].set_title('Line Chart: Sepal Length Trend')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Sepal Length (cm)')
axes[0, 0].grid(True)

# Plot 2: Bar Chart
# Visualize the average sepal length for each species, which we calculated earlier.
species_mean.plot(kind='bar', ax=axes[0, 1], color=['skyblue', 'lightgreen', 'salmon'])
axes[0, 1].set_title('Bar Chart: Average Sepal Length by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Average Sepal Length (cm)')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Histogram
# Display the distribution of 'petal length (cm)'.
axes[1, 0].hist(df['petal length (cm)'], bins=10, color='skyblue', edgecolor='black')
axes[1, 0].set_title('Histogram: Petal Length Distribution')
axes[1, 0].set_xlabel('Petal Length (cm)')
axes[1, 0].set_ylabel('Frequency')

# Plot 4: Scatter Plot
# Visualize the relationship between 'sepal length' and 'petal length'.
axes[1, 1].scatter(df['sepal length (cm)'], df['petal length (cm)'], c=iris.target, cmap='viridis')
axes[1, 1].set_title('Scatter Plot: Sepal Length vs. Petal Length')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')

# Use plt.tight_layout() to ensure all plot elements are visible and don't overlap.
plt.tight_layout()

# Display the plots.
plt.show()
