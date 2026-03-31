import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv("iris.csv")


X = df.drop("species", axis=1)


wcss = []

K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.plot(K_range, wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()
