#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from tf.transformations import quaternion_matrix
from sensor_msgs import point_cloud2

import numpy as np
import cv2 as cv
import csv
from matplotlib import pyplot as plt
from collections import Counter

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2


class ObjectRecognitionNode(object):
    def __init__(self):
        self._tf_buffer = tf2_ros.Buffer(
            rospy.Duration(1200.0))  # tf buffer length
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._camera_input_image_raw_topic = rospy.get_param("~camera_input_image_raw_topic",
                                                             "/rgb/image_color")
        self._camera_input_camera_info_topic = rospy.get_param("~camera_input_info",
                                                               "/rgb/camera_info")
        self._tagged_image_output_topic = rospy.get_param("~tagged_image_output_topic",
                                                          "/rgb/image_color_tagged")
        self._point_cloud_data_topic = rospy.get_param("~pointcloud_data",
                                                       "/depth/points")

        self._depth_sensor_min_distance = rospy.get_param(
            "~depth_sensor_min_distance", 0.3)
        self._depth_sensor_max_distance = rospy.get_param(
            "~depth_sensor_max_distance", 2.0)

        self._canny_threshold_min = rospy.get_param(
            "~canny_threshold_min", 100)
        self._canny_threshold_max = rospy.get_param(
            "~canny_threshold_max", 500)

        # don't let anything uninitialized
        self._image_mask = None
        self._camera_model = None
        self._directions_of_interest = None
        self._directions_of_interest_timetag = None
        self._directions_of_interest_frame = None

        self._cv_bridge = CvBridge()

        # publishers
        self._tagged_image_output_pub = rospy.Publisher(
            self._tagged_image_output_topic, Image, queue_size=1)
        # subscribers
        self._image_raw_sub = rospy.Subscriber(
            self._camera_input_image_raw_topic, Image, self._image_raw_callback, queue_size=1)
        self._camera_info_sub = rospy.Subscriber(
            self._camera_input_camera_info_topic, CameraInfo, self._camera_info_callback)
        self._camera_info_sub = rospy.Subscriber(
            self._point_cloud_data_topic, PointCloud2, self._point_cloud_data_callback,  queue_size=1)

    def _camera_info_callback(self, camera_info_msg):
        if not self._camera_model:
            self._camera_model = PinholeCameraModel()
            self._camera_model.fromCameraInfo(camera_info_msg)
            rospy.loginfo("Got camera description message:")
            rospy.loginfo(" K = {0}, D = {1}".format(
                self._camera_model.intrinsicMatrix(),
                self._camera_model.distortionCoeffs()))

    def _remove_phantom_objects(self, contours, rectangles, areas, mean_colors):
        valid_indexes = []
        for i in range(len(rectangles)):
            rect = rectangles[i]
            x1, y1, w, h = rect
            x2, y2 = x1+w-1, y1+h-1
            m = self._image_mask
            # if any of the corners fall within the mask, ignore this item
            if m[y1, x1] != 0 or m[y1, x2] != 0 or m[y2, x2] != 0 or m[y2, x1] != 0:
                continue
            valid_indexes.append(i)
        filtered_contours = [contours[index] for index in valid_indexes]
        filtered_rectangles = [rectangles[index] for index in valid_indexes]
        filtered_areas = [areas[index] for index in valid_indexes]
        filtered_mean_colors = [contours[index] for index in valid_indexes]
        return filtered_contours, filtered_rectangles, filtered_areas, filtered_mean_colors

    def _image_raw_callback(self, image_msg):
        # Convert the image from ros msg to opencv domain
        input_cv_image = self._cv_bridge.imgmsg_to_cv2(
            image_msg, desired_encoding='rgb8')
        # Create the mask only once, then reuse
        if self._image_mask is None:
            self._image_mask = self._create_view_for_image(input_cv_image)
        # process image to detect objects
        contours, rectangles, areas, mean_colors = self._get_image_contours(
            input_cv_image, (self._canny_threshold_min, self._canny_threshold_max))
        # Apply the mask
        contours, rectangles, areas, mean_colors = self._remove_phantom_objects(
            contours, rectangles, areas, mean_colors)

        # determine the direction of interesting objects
        self._directions_of_interest = self._determine_interesting_vectors(
            rectangles)
        self._directions_of_interest_timetag = image_msg.header.stamp
        self._directions_of_interest_frame_id = image_msg.header.frame_id
        # create output image by tagging known information in it
        output_cv_image = input_cv_image.copy()
        # add information recovered through detection to the image
        self._tag_image_detections_on_output_image(
            output_cv_image, contours, rectangles, areas, mean_colors)
        # Convert the image from opencv domain to ros msg
        output_image = self._cv_bridge.cv2_to_imgmsg(
            output_cv_image, encoding='rgb8')
        # Publish information
        self._tagged_image_output_pub.publish(output_image)

    def _camera_info_callback(self, camera_info_msg):
        if not self._camera_model:
            self._camera_model = PinholeCameraModel()
            self._camera_model.fromCameraInfo(camera_info_msg)
            rospy.loginfo("Got camera description message:")
            rospy.loginfo(" K = {0}, D = {1}".format(
                self._camera_model.intrinsicMatrix(),
                self._camera_model.distortionCoeffs()))

    def _determine_distance_to_points(self, point_cloud_2_msg, directions_of_interest):
        # Object count will be used to generate unique ids for clusters
        best_product = [-1.0] * len(directions_of_interest)
        best_distance = [None] * len(directions_of_interest)

        np_points = np.asarray(list(point_cloud2.read_points(point_cloud_2_msg, skip_nans=True)))
        distances = np.sqrt(np.sum(np_points**2, axis=1))
        hit_directions = np_points / distances.reshape(76800, 1)

        indexes = np.all([distances > self._depth_sensor_min_distance, distances < self._depth_sensor_max_distance], axis=0)
        np_points = np_points[indexes,:]
        distances = distances[indexes]
        hit_directions = hit_directions[indexes, :]

        for vector_index in range(len(directions_of_interest)):
            dot_products = np.abs(np.sum(hit_directions * np.asarray(directions_of_interest[vector_index]), axis=1))
            for sample_index in range(len(distances)):       
                if (best_product[vector_index] < dot_products[sample_index]):
                    best_product[vector_index] = dot_products[sample_index]
                    best_distance[vector_index] = distances[sample_index]
        
        found_objects = []
        for i in range(len(directions_of_interest)):
            # ignore object we did no find the distance to
            if best_product[i] < 0.0:
                rospy.logerr("We did not find the distance in the direction of %s" % (
                    str(directions_of_interest[i])))
                continue
            found_objects.append(i)
            rospy.loginfo("Object through %s at %f meters!" %            (str(directions_of_interest[i]), best_distance[i]))
        return [{"direction": directions_of_interest[i], "distance":best_distance[i]} for i in found_objects]

    def _point_cloud_data_callback(self, point_cloud_2_msg):
        # If there are no directions of interest, do nothing
        if self._directions_of_interest is None:
            return
        # Convert the pointcloud2 data to a list of detected clusters
        # with coordinates, so that we deal with a smaller number of measurements
        detected_objects = self._determine_distance_to_points(
            point_cloud_2_msg,
            self._directions_of_interest
        )

    def _tag_image_detections_on_output_image(self, output_cv_image, contours, rectangles, areas, mean_colors):
        for _, rect, areas, color in zip(contours, rectangles, areas, mean_colors):
            x1, y1, w, h = rect
            x2, y2 = x1+w-1, y1+h-1
            cv.rectangle(output_cv_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    def _determine_interesting_vectors(self, rectangles):
        if self._camera_model is None:
            return
        vectors = []
        for rect in rectangles:
            x, y, w, h = rect
            xc, yc = x + w/2, y + h/2
            vectors.append(self._camera_model.projectPixelTo3dRay((xc, yc)))
        return vectors

    def _create_view_for_image(self, input_image):
        mask = np.zeros(input_image.shape[:2], np.uint8)
        height, width, _ = input_image.shape
        y1 = int(height * 0.80)
        cv.rectangle(mask, (0, y1), (width-1, height-1), 255, -1)
        cv.rectangle(mask, (0, 0), (10, 10), 255, -1)
        cv.rectangle(mask, (width-11,0), (10, 10), 255, -1)
        return mask

    def _get_image_contours(self, input_image, canny_args=(300, 500)):
        edges = cv.Canny(input_image, canny_args[0], canny_args[1])
        blurred = cv.GaussianBlur(edges, (3, 9), 0)
        _, contours, __ = cv.findContours(
            blurred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bounding_rectangles = [cv.boundingRect(cnt) for cnt in contours]
        contour_areas = [cv.contourArea(cnt) for cnt in contours]
        mean_colors = []
        for cnt in contours:
            mask = np.zeros(input_image.shape[:2], np.uint8)
            cv.drawContours(mask, [cnt], 0, 255, -1)
            mean_color = cv.mean(input_image, mask=mask)
            mean_colors.append(mean_color[:3])
        return contours, bounding_rectangles, contour_areas, mean_colors

    def _convert_location_between_frames(self, source_frame, source_timestamp, dest_frame, dest_timestamp, location):

        transform = self._tf_buffer.lookup_transform_full(target_frame=dest_frame,
                                                          target_time=dest_timestamp,
                                                          source_frame=source_frame,
                                                          source_time=source_timestamp,
                                                          fixed_frame="cora/odom",
                                                          timeout=rospy.Duration(
                                                              1.0)
                                                          )
        original = PointStamped()
        original.header.frame_id = source_frame
        original.header.stamp = source_timestamp
        original.point = location
        pose = tf2_geometry_msgs.do_transform_point(original, transform)
        return Point(x=pose.point.x, y=pose.point.y, z=pose.point.z)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('object_recognition_node', anonymous=True)
    ObjectRecognitionNode().run()
