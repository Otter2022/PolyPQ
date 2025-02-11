import PolyPQ

# For example:
points = [
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0]
]
centers = [
    [1.0, 2.0],
    [5.0, 8.0]
]

# Using the default (euclidean) metric:
clusters = PolyPQ.kmeans(points, centers, 100)
print("Euclidean clusters:", clusters)

# Using the jaccard metric:
clusters_jaccard = PolyPQ.kmeans(points, centers, 100, metric="jaccard")
print("Jaccard clusters:", clusters_jaccard)
