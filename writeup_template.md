## Project: Perception Pick & Place

---

# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 

I used the sensor_stick project source code for training an SVM classifier based on normal and color histogram features. Here are the objects and training results for each test world. I increased the number of training samples per object to 50.


**Test world 1**:

```python
models = [
    'biscuits',
    'soap',
    'soap2',
]
```

![model_1_confusion](/images/model_1_confusion.png)
![model_1_confusion_normal](/images/model_1_confusion_normal.png)


**Test world 2**:

```python
models = [
    'biscuits',
    'soap',
    'soap2',
    'book',
    'glue',
]
```

![model_2_confusion](/images/model_2_confusion.png)
![model_2_confusion_normal](/images/model_2_confusion_normal.png)


**Test world 3**:

```python
models = [
    'biscuits',
    'soap',
    'soap2',
    'book',
    'glue',
    'sticky_notes',
    'snacks',
    'eraser',
]
```

![model_3_confusion](/images/model_3_confusion.png)
![model_3_confusion_normal](/images/model_3_confusion_normal.png)


Overall, the training results look satisfactory for all three test worlds.


2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.

```python
pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
```

3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.

In order to reduce noise, I used PCL's Statistical Outlier Filtering function with the following parameters.

```python
outlier_filter = cloud_filtered.make_statistical_outlier_filter()
outlier_filter.set_mean_k(50)
outlier_filter.set_std_dev_mul_thresh(0.1)
cloud_filtered = outlier_filter.filter()
```

After that, I applied VoxelGrid downsampling, z-thresholding and RANSAC plane fitting to extract the table top:

```python
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

# TODO: RANSAC Plane Segmentation
seg = cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(.01)
inliers, coefficients = seg.segment()

# TODO: Extract inliers and outliers
cloud_table = cloud_filtered.extract(inliers)
cloud_objects = cloud_filtered.extract(inliers, negative=True)
```

4. Apply Euclidean clustering to create separate clusters for individual items.

```python
# TODO: Euclidean Clustering
white_cloud = XYZRGB_to_XYZ(cloud_objects)
tree = white_cloud.make_kdtree()
ec = white_cloud.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(.02)
ec.set_MinClusterSize(100)
ec.set_MaxClusterSize(10000)
ec.set_SearchMethod(tree)
cluster_indices = ec.Extract()
```

5. Perform object recognition on these objects and assign them labels (markers in RViz).

```python
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
```

6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.

```python
for do in detected_objects:
    # TODO: Get the PointCloud for a given object and obtain it's centroid
    labels.append(do.label)
    points_arr = ros_to_pcl(do.cloud).to_array()
    centroid = np.mean(points_arr, axis=0)[:3].tolist()
    centroids.append(centroid)
```

7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  

```python
for do in detected_objects:
        ...

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
```

8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.

Output files are in [pr2_robot/scripts](https://github.com/aghagol/RoboND-Perception-Project/tree/master/pr2_robot/scripts) directory.

9. Congratulations!  Your Done!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

Completed Excercise 1 code can be found here: https://github.com/aghagol/RoboND-Perception-Exercises/blob/master/Exercise-1/RANSAC.py

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

Completed Excercise 2 code can be found here: https://github.com/aghagol/RoboND-Perception-Exercises/blob/master/Exercise-2/sensor_stick/scripts/segmentation.py

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

Completed Excercise 3 code can be found here: https://github.com/aghagol/RoboND-Perception-Exercises/blob/master/Exercise-3/sensor_stick/scripts/object_recognition.py

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

The steps for object recognition for all 3 test worlds are described above.

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  



