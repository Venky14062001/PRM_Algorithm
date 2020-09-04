import PRM_Algorithm_Functions as prm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.spatial import Delaunay

# Create a Delaunay object to plot obstacle traingle
list_point=[]
list_delaunay=[]
for i in range(len(prm.env.obs)):
    list_point.append(np.array([[prm.env.obs[i].x0,prm.env.obs[i].y0],[prm.env.obs[i].x1,prm.env.obs[i].y1],[prm.env.obs[i].x2,prm.env.obs[i].y2]]))
    list_delaunay.append(Delaunay(list_point[i]))

# Create a Pandas Dataframe consisting of a given number of random points in the environment object
csv=prm.list_random_points(prm.env, 1000)
 
# Generate a sorted cluster dataframe
sorted_cluster_df=prm.create_clusters(csv,50)
sorted_cluster_df.rename({0: 'x_coord', 1: 'y_coord'}, axis=1, inplace=True)

# Generate cluster point objects list
cluster_points=prm.create_cluster_point_objects(sorted_cluster_df)

# Generate obstacle point objects list
obstacle_points=prm.create_obstacle_point_objects(prm.env)

# Generate cluster center point objects list
cluster_center_points=prm.create_cluster_center_point_objects(prm.cluster_centers_list)

# Generate a list of non intersecting with the obstacle points in the clusters
required_list=prm.create_non_intersecting_cluster_point_objects(cluster_points, cluster_center_points, obstacle_points)

# Generate a list of non intersecting with the obstacles coordinates for visualisation
required_list_visualisation=prm.convert_from_non_intersecting_cluster_point_objects_to_dataframe(required_list)

# Plot the obstacles
f, axes = plt.subplots(1, 1, figsize=(16,8))
for i in range(len(prm.env.obs)):
    plt.triplot(list_point[i][:,0], list_point[i][:,1], list_delaunay[i].simplices)
    plt.plot(list_point[i][:,0], list_point[i][:,1], 'o')

# Plot the initial random points generated without collision with obstacles
f, axes = plt.subplots(1, 1, figsize=(16,8))
# Plot the obstacles
for i in range(len(prm.env.obs)):
    plt.triplot(list_point[i][:,0], list_point[i][:,1], list_delaunay[i].simplices)
    plt.plot(list_point[i][:,0], list_point[i][:,1], 'o')    
x, y = zip(*(csv))
plt.scatter(x, y)
plt.show()

# Summary of the Cluster Labels
f, axes = plt.subplots(1, 1, figsize=(16,8))
sb.countplot(sorted_cluster_df["Cluster"])

# Plot the visualisation of the cluster points
f, axes = plt.subplots(1, 1, figsize=(16,8))
# Plot the obstacles
for i in range(len(prm.env.obs)):
    plt.triplot(list_point[i][:,0], list_point[i][:,1], list_delaunay[i].simplices)
    plt.plot(list_point[i][:,0], list_point[i][:,1], 'o')   
plt.scatter(x = "x_coord", y = "y_coord", c = "Cluster", cmap = 'viridis', data = sorted_cluster_df)

# Plot the visualisation of non intersecting clusters
f, axes = plt.subplots(1, 1, figsize=(16,8))
# Plot the obstacles
for i in range(len(prm.env.obs)):
    plt.triplot(list_point[i][:,0], list_point[i][:,1], list_delaunay[i].simplices)
    plt.plot(list_point[i][:,0], list_point[i][:,1], 'o')  
required_df = pd.DataFrame(required_list_visualisation,columns=['x_coord','y_coord','Cluster'])
plt.scatter(x = "x_coord", y = "y_coord", c = "Cluster", cmap = 'viridis', data = sorted_cluster_df)


# The application of Dijkstras shortest path algorithm on these points can help find the shortest route without obstacles
# TBC
