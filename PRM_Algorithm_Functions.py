# Import required libraries
import numpy as np 
import pandas as pd 
import seaborn as sb 
from sklearn.cluster import KMeans
import matplotlib.pyplot as pl
import Intersect
import sys

# Change the current system path to import 
sys.path.append('osr_examples/scripts/')
import environment_2d


# Open a window
pl.ion()
np.random.seed(4)

# Create an Environment instance
env=environment_2d.Environment(10,6,5)

pl.clf()
env.plot()

q=env.random_query()

#List of coordinates of obstacles
obs_list=env.obs

'''Generate a random point in the environment that does not collide with obstacle'''
def generate_random_point(env_object):
    size_x=env_object.size_x
    size_y=env_object.size_y

    # Get a random point
    random_point_x=np.random.rand()*size_x
    random_point_y=np.random.rand()*size_y

    # Check if the point collides with the obstacles
    while env.check_collision(random_point_x,random_point_y):
        random_point_x=np.random.rand()*size_x
        random_point_y=np.random.rand()*size_y
    
    return (random_point_x,random_point_y)

'''Create a list of n number of random points not colliding with obstacles'''
def list_random_points(env_object, no_of_points):
    global random_points_list
    
    random_points_list=[]

    for i in range(no_of_points):
        random_point=generate_random_point(env_object)
        random_points_list.append(random_point)  
        
    return random_points_list

'''Create given number of clusters of nearest random points using KMeans Clustering'''
def create_clusters(list_of_points_csv, no_of_clusters):
    global X_labeled_sorted
    # Create a pandas dataframe
    X=pd.DataFrame(list_of_points_csv)

    # Create clustering model using KMeans
    kmeans=KMeans(n_clusters=no_of_clusters)

    # Fit the clustering model on data
    kmeans.fit(X)

    # Create a global variable of cluster centers
    global cluster_centers_list
    cluster_centers_list=kmeans.cluster_centers_

    # Predict the cluster labels
    labels=kmeans.predict(X)

    # Append labels to the data
    X_labeled=X.copy()
    X_labeled["Cluster"]=pd.Categorical(labels)
    X_labeled_sorted=X_labeled.sort_values(by="Cluster")
    return X_labeled_sorted


'''Create clusteres point objects'''
def create_cluster_point_objects(sorted_cluster_dataframe):
    global point_object_cluster_list
    # Dataframe to list
    sorted_cluster_list=sorted_cluster_dataframe.values.tolist()
    
    # Convert coordinates to point objects
    point_object_cluster_list=[]
    for i in range(len(sorted_cluster_list)):
        point_object_cluster_list.append((Intersect.Point(sorted_cluster_list[i][0],sorted_cluster_list[i][1]),sorted_cluster_list[i][2]))

    return point_object_cluster_list


'''Create point objects of the obstacles in environment'''
def create_obstacle_point_objects(env_object):
    global list_obs_point_object
    list_obs_point_object=[[] for i in range(len(env_object.obs))]

    for i in range(len(env_object.obs)):
        list_obs_point_object[i].append((Intersect.Point(env_object.obs[i].x0,env_object.obs[i].y0),Intersect.Point(env_object.obs[i].x1,env_object.obs[i].y1)))
        list_obs_point_object[i].append((Intersect.Point(env_object.obs[i].x1,env_object.obs[i].y1),Intersect.Point(env_object.obs[i].x2,env_object.obs[i].y2)))
        list_obs_point_object[i].append((Intersect.Point(env_object.obs[i].x2,env_object.obs[i].y2),Intersect.Point(env_object.obs[i].x0,env_object.obs[i].y0)))

    return list_obs_point_object


'''Create point objects for cluster centers'''
def create_cluster_center_point_objects(cluster_centers):
    global list_cluster_center_point_object
    list_cluster_center_point_object=[]

    for i in range(len(cluster_centers)):
        list_cluster_center_point_object.append(Intersect.Point(cluster_centers[i][0],cluster_centers[i][1]))

    return list_cluster_center_point_object


'''Fuction to check if the line joining cluster centers to the points in that respective cluster intersects the obstacles, if yes, then reject that point in the cluster'''
def create_non_intersecting_cluster_point_objects(cluster_points, cluster_center_points, obstacle_points):
    global list_non_intersecting_cluster_point_objects
    list_non_intersecting_cluster_point_objects=[]

    for i in range(len(cluster_points)):

        # Determining the cluster number
        cluster_number=cluster_points[i][1]

        for j in range(len(obstacle_points)):
            # Variable to store if there is intersection
            intersect=False
            for k in range(len(obstacle_points[j])):
                if not Intersect.doIntersect(cluster_points[i][0],cluster_center_points[cluster_number],obstacle_points[j][k][0],obstacle_points[j][k][1]):
                    pass
                else:
                    intersect=True

        if not intersect:
            list_non_intersecting_cluster_point_objects.append(cluster_points[i])

    return list_non_intersecting_cluster_point_objects

'''Function to convert the non intersecting point objects to a list'''
def convert_from_non_intersecting_cluster_point_objects_to_dataframe(list_non_intersecting_points):
    global list_non_intersecting_cluster
    
    # Create a list to contain the cluster coordinates
    list_non_intersecting_cluster=[]
    
    for i in range(len(list_non_intersecting_points)):
        list_non_intersecting_cluster.append((list_non_intersecting_points[i][0].x,list_non_intersecting_points[i][0].y,list_non_intersecting_points[i][1]))
        
    return list_non_intersecting_cluster
