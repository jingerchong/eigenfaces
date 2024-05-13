from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=5, resize=0.4)

# Get the data and target
X = lfw_people.data
y = lfw_people.target

# Calculate the number of faces per person
unique, counts = np.unique(y, return_counts=True)

# Calculate Q1 (25th percentile) and Q3 (75th percentile) of faces per person
Q1 = np.percentile(counts, 25)
Q3 = np.percentile(counts, 75)

# Calculate the IQR
IQR = Q3 - Q1

# Set a multiplier for the IQR (e.g., 1.5)
multiplier = 1.5

# Calculate the lower and upper bounds
lower_bound = Q1 - multiplier * IQR
upper_bound = Q3 + multiplier * IQR

# Identify outliers (persons with more faces than the upper bound)
outliers = []
outliers_count = []
inliers = []
inliers_count = []
for person_id, num_faces in zip(unique, counts):
    if num_faces > upper_bound or num_faces < lower_bound:
        outliers.append(person_id)
        outliers_count.append(num_faces)
    else:
        inliers.append(person_id)
        inliers_count.append(num_faces)
        
# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(inliers, inliers_count, marker='.',)
plt.scatter(outliers, outliers_count, color='red', marker='x', label='outlier')

plt.xlabel('Person ID')
plt.ylabel('Number of Faces')

# Add lines for the region of the IQR
plt.axhline(y=Q1, color='r', linestyle='--', label='Q1')
plt.axhline(y=Q3, color='b', linestyle='--', label='Q3')
plt.axhline(y=lower_bound, color='g', linestyle='--', label='lower bound (Q1 - 1.5 * IQR)')
plt.axhline(y=upper_bound, color='purple', linestyle='--', label='upper bound (Q3 + 1.5 * IQR)')

plt.legend()
plt.savefig(os.path.join('diagrams', f'lfw_outliers.png'))