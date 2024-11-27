#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class LineFollower:
    def __init__(self):
        # Initialize the node
        rospy.init_node('line_follower_bot', anonymous=True)

        # Publisher for robot movement
        self.pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Subscriber for the image topic
        self.image_sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback)

        # Initialize Twist message for movement commands
        self.cmd_move = Twist()

    def image_callback(self, msg):
        """
        Callback function for processing incoming image messages.
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Crop to the bottom half of the image for better road detection
        height, width, _ = cv_image.shape
        cropped_image = cv_image[height // 2:, :]

        # Convert to grayscale
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold the image to detect white lines
        _, threshold_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY)

        # Find contours of the white lines
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Identify the two largest contours corresponding to the white lines
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            centroids = []

            for contour in sorted_contours:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    centroids.append(cx)

            if len(centroids) == 2:
                # Calculate the midpoint between the two white lines
                lane_center = (centroids[0] + centroids[1]) // 2
                image_center = width // 2
                error = lane_center - image_center

                # Proportional controller for angular velocity
                k_p = 0.004
                angular_z = -k_p * error

                # Adjust linear speed based on the error
                max_linear_speed = 1.0
                min_linear_speed = 0.5
                linear_speed = max_linear_speed - (abs(error) / (width // 2)) * (max_linear_speed - min_linear_speed)
                linear_speed = max(min_linear_speed, linear_speed)  # Ensure the speed doesn't go below the minimum

                # Update Twist message with new speeds
                self.cmd_move.linear.x = linear_speed
                self.cmd_move.angular.z = angular_z

                # Publish the command
                self.pub.publish(self.cmd_move)
            else:
                rospy.logwarn("Could not detect both white lines. Slowing down.")
                self.cmd_move.linear.x = 0.0
                self.cmd_move.angular.z = 0.2  # Rotate slightly to search for the lines
                self.pub.publish(self.cmd_move)
        else:
            rospy.logwarn("No lines detected. Stopping.")
            self.cmd_move.linear.x = 0.0
            self.cmd_move.angular.z = 0.3  # Rotate to search for the lines
            self.pub.publish(self.cmd_move)

    def run(self):
        # Keep the node running
        rospy.spin()

if __name__ == '__main__':
    try:
        line_follower = LineFollower()
        line_follower.run()
    except rospy.ROSInterruptException:
        pass
