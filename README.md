# PRM_Algorithm

Steps of algorithm

1. Generate n random samples in the environment object
2. Check if the samples collide with the given obstacles
3. Find k neighbours/clusters using KNN nearest neighbour:
     -> Link each cluster center with the cluster points
     -> Check if the line between cluster center and cluster points intersects with the obstacle, if yes, discard that cluster point
4. Search for shortest path from start to end node using the Dijksta's shortest path algorithm.
