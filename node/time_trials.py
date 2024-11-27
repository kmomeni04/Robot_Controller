#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

MODULE_NAME = "Driving Module"


def fprint(*args):
    """
    Custom print function that adds the module name to any print statements.
    """
    print(str(rospy.get_rostime().secs) + ": " + MODULE_NAME + ": " + " ".join(map(str, args)))


class RobotDriver:
    WHITE_THRESHOLD = 200  # Threshold for detecting white pixels
    STOP_LINE_THRESHOLD = 5000  # Threshold for detecting stop lines
    OBSTACLE_AREA_THRESHOLD = 2000  # Minimum area to consider as an obstacle

    def __init__(self):
        # Initialize ROS node
        rospy.init_node('robot_driver', anonymous=True)

        # Publisher for robot velocity commands
        self.cmd_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)

        # Publisher for timer
        self.timer_pub = rospy.Publisher('/score_tracker', String, queue_size=1)

        # Subscriber for the camera image topic
        self.image_sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback)

        # Initialize CvBridge for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Command to control robot velocity
        self.cmd = Twist()

        # State tracking
        self.searching_for_line = False
        self.start_time = rospy.get_rostime().secs

    def image_callback(self, msg):
        try:
            # Convert the ROS image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Process the image to follow the road and avoid obstacles
        self.process_image(cv_image)

        # Publish the elapsed time
        elapsed_time = rospy.get_rostime().secs - self.start_time
        if elapsed_time <= 240:
            self.timer_pub.publish(String(data=f"sherlock,detective,{elapsed_time},AAAA"))
        else:
            rospy.loginfo("240 seconds have passed. Stopping score tracking.")

    def process_image(self, img):
        # Convert the image to grayscale using the green channel
        img[:, :, 0] = img[:, :, 1]  # Remove the Red channel
        img[:, :, 2] = img[:, :, 1]  # Remove the Blue channel
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Crop to the bottom half of the image
        height, width = gray.shape
        cropped_gray = gray[height // 2:height, :]

        # Resize the cropped image
        shape = cropped_gray.shape
        resized_gray = cv2.resize(cropped_gray, (int(shape[1] / 2), int(shape[0] / 2)), interpolation=cv2.INTER_CUBIC)

        # Apply Gaussian Blur to reduce noise
        blurred_gray = cv2.GaussianBlur(resized_gray, (5, 5), 0)

        # Threshold the image to detect white lines
        _, thresholded = cv2.threshold(blurred_gray, self.WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Find all white pixels
        white_pixels = np.column_stack(np.where(thresholded == 255))

        # Calculate the left and right boundaries of the white lines
        if len(white_pixels) > 0:
            left_pixels = white_pixels[white_pixels[:, 1] < width // 2]
            right_pixels = white_pixels[white_pixels[:, 1] >= width // 2]

            if len(left_pixels) > 0 and len(right_pixels) > 0:
                left_boundary = np.min(left_pixels[:, 1])
                right_boundary = np.max(right_pixels[:, 1])
                lane_center = (left_boundary + right_boundary) // 2
                center = width // 2
                error = lane_center - center

                # Proportional, integral, and derivative control for steering
                kp = 0.005
                kd = 0.002
                ki = 0.001

                # Calculate the orientation of the white lines using linear regression
                vx, vy, x_mean, y_mean = cv2.fitLine(white_pixels, cv2.DIST_L2, 0, 0.01, 0.01)
                slope = vy / vx

                # Calculate the desired steering angle based on the line orientation and error
                angle_correction = kp * error - kd * slope
                self.cmd.angular.z = angle_correction
                self.cmd.linear.x = 0.4  # Reduced constant forward speed to improve turning
                self.searching_for_line = False
            else:
                rospy.logwarn("Unable to detect both lane boundaries.")
                self.cmd.linear.x = 0.0
                self.cmd.angular.z = 0.3
                self.searching_for_line = True
        else:
            # If no white pixels are detected, stop or rotate to find the line
            rospy.logwarn("No line detected, searching...")
            if not self.searching_for_line:
                self.cmd.linear.x = 0.0
                self.cmd.angular.z = 0.3
                self.searching_for_line = True

        # Obstacle detection using contour detection (detecting shapes)
        edges = cv2.Canny(blurred_gray, 50, 150)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.OBSTACLE_AREA_THRESHOLD:
                # Calculate the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # Check if the detected shape resembles an obstacle (e.g., vertical barriers)
                if 0.5 < aspect_ratio < 2.0 and y + h > height * 0.3:
                    rospy.loginfo("Obstacle detected on the road: Stopping the robot.")
                    self.cmd.linear.x = 0.0
                    self.cmd.angular.z = 0.0
                    break
        else:
            # Resume normal speed if no obstacle is detected
            if len(white_pixels) > 0:
                self.cmd.linear.x = 0.4

        # Adjust steering to ensure the robot stays parallel to the lane
        if len(white_pixels) > 0:
            left_edge = np.percentile(left_pixels[:, 1], 25) if len(left_pixels) > 0 else 0
            right_edge = np.percentile(right_pixels[:, 1], 75) if len(right_pixels) > 0 else width
            lane_width = right_edge - left_edge
            desired_distance_from_left = lane_width * 0.5
            current_distance_from_left = lane_center - left_boundary if len(left_pixels) > 0 else 0

            lateral_error = desired_distance_from_left - current_distance_from_left

            kp_lateral = 0.01
            lateral_correction = kp_lateral * lateral_error
            self.cmd.angular.z += lateral_correction

        # Publish the command to move the robot
        self.cmd_pub.publish(self.cmd)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz rate for control
        fprint("Starting Driving Module")

        # Publish the initial score
        self.timer_pub.publish(String(data='sherlock,detective,0,AAAA'))

        while not rospy.is_shutdown():
            # Publish the command and timer at a consistent rate
            self.cmd_pub.publish(self.cmd)
            elapsed_time = rospy.get_rostime().secs - self.start_time
            self.timer_pub.publish(String(data=f"sherlock,detective,{elapsed_time},AAAA"))
            rate.sleep()

        # On shutdown, stop the robot
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        self.cmd_pub.publish(self.cmd)
        fprint("Stopping Driving Module")


if __name__ == '__main__':
    try:
        driver = RobotDriver()
        driver.run()
    except rospy.ROSInterruptException:
        pass
