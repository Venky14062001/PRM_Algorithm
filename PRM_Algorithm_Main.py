import PRM_Algorithm_Functions as prm

# Create a Pandas Dataframe consisting of a given number of random points in the environment object
csv=prm.list_random_points(prm.env, 1000)
 
# Generate cluster point objects list
cluster_points=prm.create_cluster_point_objects(prm.create_clusters(csv,50))

# Generate obstacle point objects list
obstacle_points=prm.create_obstacle_point_objects(prm.env)

# Generate cluster center point objects list
cluster_center_points=prm.create_cluster_center_point_objects(prm.cluster_centers_list)

# Generate a list of non intersecting with the obstacle points in the clusters
required_list=prm.create_non_intersecting_cluster_point_objects(cluster_points, cluster_center_points, obstacle_points)

# Print out the point objects
print(required_list)

# The application of Dijkstras shortest path algorithm on these points can help find the shortest route without obstacles
