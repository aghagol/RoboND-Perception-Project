#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud_filtered = ros_to_pcl(pcl_msg)

    # TODO: Statistical Outlier Filtering
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50)
    outlier_filter.set_std_dev_mul_thresh(0.1)
    cloud_filtered = outlier_filter.filter()

    # TODO: Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = .01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # TODO: PassThrough Filter to extract table top and objects
    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name('z')
    passthrough.set_filter_limits(0.608, 0.8)
    cloud_filtered = passthrough.filter()

    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name('y')
    passthrough.set_filter_limits(-.5, .5)
    cloud_filtered = passthrough.filter()

    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(.01)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(.02)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(10000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color_list = get_color_list(len(cluster_indices))
    cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, idx in enumerate(indices):
            cluster_point_list.append([
                white_cloud[idx][0],
                white_cloud[idx][1],
                white_cloud[idx][2],
                rgb_to_float(cluster_color_list[j])])
    cloud_cluster = pcl.PointCloud_PointXYZRGB()
    cloud_cluster.from_list(cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_cluster = pcl_to_ros(cloud_cluster)

    # TODO: Publish ROS messages
    pcl_table_pub.publish(ros_cloud_table)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_cluster_pub.publish(ros_cloud_cluster)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)

        # Compute the associated feature vector
        ros_cluster = pcl_to_ros(pcl_cluster)
        chists = compute_color_histograms(ros_cluster, using_hsv=False)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(
        len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(detected_objects):

    # TODO: Initialize variables
    test_scene_num = Int32()
    test_scene_num.data = 1

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # TODO: Parse parameters into individual variables
    label_to_group = {object_item['name']: object_item['group']
        for object_item in object_list_param}

    group_to_position = {dropbox_item['group']: dropbox_item['position']
        for dropbox_item in dropbox_param}

    group_to_arm = {dropbox_item['group']: dropbox_item['name']
        for dropbox_item in dropbox_param}

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    yaml_dict_list = []

    for do in detected_objects:
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        labels.append(do.label)
        points_arr = ros_to_pcl(do.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3].tolist()
        centroids.append(centroid)

        # TODO: Create 'place_pose' for the object
        object_name = String()
        object_name.data = str(do.label)

        pick_pose = Pose()
        pick_pose.position.x = centroid[0]
        pick_pose.position.y = centroid[1]
        pick_pose.position.z = centroid[2]

        place_pose = Pose()
        place_pose.position.x = group_to_position[label_to_group[do.label]][0]
        place_pose.position.y = group_to_position[label_to_group[do.label]][1]
        place_pose.position.z = group_to_position[label_to_group[do.label]][2]

        # TODO: Assign the arm to be used for pick_place
        arm_name = String()
        arm_name.data = group_to_arm[label_to_group[do.label]]

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        yaml_dict_list.append(yaml_dict)

        # # Wait for 'pick_place_routine' service to come up
        # rospy.wait_for_service('pick_place_routine')

        # try:
        #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        #     # TODO: Insert your message variables to be sent as a service request
        #     resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

        #     print ("Response: ",resp.success)

        # except rospy.ServiceException, e:
        #     print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml('output_3.yaml', yaml_dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_table_pub           = rospy.Publisher("/pcl_table",         PointCloud2,            queue_size=1)
    pcl_cluster_pub         = rospy.Publisher("/pcl_cluster",       PointCloud2,            queue_size=1)
    pcl_objects_pub         = rospy.Publisher("/pcl_objects",       PointCloud2,            queue_size=1)
    object_markers_pub      = rospy.Publisher("/object_markers",    Marker,                 queue_size=1)
    detected_objects_pub    = rospy.Publisher("/detected_objects",  DetectedObjectsArray,   queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model_3.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
