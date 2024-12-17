#! /usr/bin/env python3
import rospy
import numpy as np
import datetime
import time
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image
from std_msgs.msg import String
import tf.transformations as tft
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import time_trials1 as tt1
import math


class NavigationController():
    """
    NavigationController is responsible for managing the autonomous navigation of the robot across various operational zones. 
    It leverages computer vision techniques to detect specific color-based markers and features within the environment, 
    processes these visual inputs to determine alignment and positioning, and controls the robot's movement accordingly. 
    
    Key Functionalities:
        - **HSV Mask Creation**: Converts camera frames to the HSV color space and generates binary masks to isolate 
          specific colors (e.g., brown, purple) used for navigation cues.
        - **Contour Processing**: Identifies and filters contours from binary masks to detect significant objects or lines 
          that guide the robot's path.
        - **Zone Navigation**: Implements zone-specific navigation strategies, adjusting the robot's movement based on 
          detected features and alignment errors within each zone.
        - **PID Control**: Utilizes a Proportional-Integral-Derivative (PID) controller to fine-tune the robot's alignment, 
          ensuring smooth and accurate navigation through tunnels and around obstacles.
        - **Object Detection**: Detects special objects (e.g., "Yoda") within the environment and handles appropriate responses 
          such as pausing navigation or changing operational modes.
        - **Velocity Control**: Publishes velocity commands to control the robot's linear and angular movements through ROS 
          publishers.
        - **Score Tracking**: Maintains and updates a score tracker to monitor the robot's progress and performance during missions.
    
    Integration with ROS:
        - Subscribes to relevant ROS topics to receive sensor data and publishes control commands to navigate the robot effectively.
        - Interfaces with ROS message types like `Twist` for velocity commands and `String` for score tracking.
    
    User Interface:
        - Displays processed masks and edge-detected images on the GUI for debugging and monitoring purposes.
    
    Overall, the NavigationController ensures that the robot can autonomously navigate complex environments by intelligently 
    interpreting visual data, making real-time adjustments, and maintaining robust control over its movements.
    """
    def __init__(self, ui_interface):
        # Initialize GUI app and ROS nodes
        self.ui_interface = ui_interface

        rospy.init_node('navigation_node', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.score_publisher = rospy.Publisher('/score_tracker', String, queue_size=1)

        self.image_converter = CvBridge()
        rospy.Subscriber('/B1/camera1/image_raw', Image, self.handle_image_input)

        self.secondary_camera_sub = rospy.Subscriber('/B1/camera2/image_raw', Image, self.handle_secondary_image_input)
        self.last_camera_unsubscribe_time = rospy.Time.now()

        # State variables
        self.is_auto_navigation = False
        self.should_save_image = False
        self.should_update_score = True
        self.navigation_start_time = 0
        self.processed_frame_count = 0
        self.last_status_update_time = None

        self.initialize_state()


    def initialize_state(self):
        self.current_zone = 1
        self.last_zone_transition_time = None
        self.last_clue_timestamp = None
        self.zone3_phase = 'hill'
        self.zone4_phase = 'tunnel'
        self.last_highres_frame = None
        self.clue_count = 1
        self.zone_confidence_score = 0

    def start_navigation(self):
        self.navigation_start_time = rospy.get_time()
        print('Auto navigation started')
        self.is_auto_navigation = True
        if self.should_update_score:
            self.update_score_tracker(0)

    def stop_navigation(self):
        print('Manual override - auto navigation stopped')
        if self.is_auto_navigation:
            self.publish_velocity(0, 0)
        self.is_auto_navigation = False
        if self.should_update_score:
            self.update_score_tracker(-1)

    def respawn_robot(self, position):
        print("Respawning robot...")
        self.publish_velocity(0, 0)
        respawn_message = ModelState()
        respawn_message.model_name = 'B1'
        respawn_message.pose.position.x = position[0]
        respawn_message.pose.position.y = position[1]
        respawn_message.pose.position.z = position[2]

        quaternion = tft.quaternion_from_euler(position[3], position[4], position[5])
        respawn_message.pose.orientation.x = quaternion[0]
        respawn_message.pose.orientation.y = quaternion[1]
        respawn_message.pose.orientation.z = quaternion[2]
        respawn_message.pose.orientation.w = quaternion[3]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state_service(respawn_message)
            print("Robot respawned successfully.")
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")

    def handle_image_input(self, image_data):
        try:
            process_start_time = time.time_ns()
            frame = self.image_converter.imgmsg_to_cv2(image_data, "bgr8")
            frame_height, frame_width = frame.shape[0:2]
            output_frame = frame.copy()
            self.processed_frame_count += 1

            # Save image if requested
            if self.should_save_image:
                filename = f'images/frame_{datetime.datetime.now().strftime("%m%d_%H:%M:%S")}.png'
                cv.imwrite(filename, frame)
                print(f'Image saved as {filename}')
                self.should_save_image = False

            # Auto navigation functionality
            if self.is_auto_navigation:
                if self.current_zone in [1, 2]:
                    # Improved line detection
                    lower_threshold = np.array([100, 50, 50])
                    upper_threshold = np.array([130, 255, 255])
                    self.detect_navigation_line(frame, lower_threshold, upper_threshold)

                    # Check for zone transition using pink lines
                    if self.detect_transition_line(frame):
                        self.current_zone += 1
                        print(f"Transitioning to Zone {self.current_zone}")

                elif self.current_zone == 3:
                    # Hill detection and line navigation
                    if self.zone3_phase == 'hill':
                        self.detect_hill_path(frame)
                        if self.detect_transition_line(frame):
                            self.zone3_phase = 'line'
                            print("Switching to line navigation in Zone 3")
                    elif self.zone3_phase == 'line':
                        lower_threshold = np.array([150, 0, 150])
                        upper_threshold = np.array([255, 100, 255])
                        self.detect_navigation_line(frame, lower_threshold, upper_threshold)
                        if self.detect_transition_line(frame):
                            self.current_zone += 1
                            print(f"Transitioning to Zone {self.current_zone}")

                elif self.current_zone == 4:
                    if self.zone4_phase == 'tunnel':
                        self.align_with_tunnel(frame)
                        if rospy.get_time() - self.navigation_start_time > 2:
                            self.zone4_phase = 'mountain'
                            print("Switching to mountain navigation in Zone 4")
                    elif self.zone4_phase == 'mountain':
                        self.follow_mountain_path(frame)
                        if self.detect_final_sign(frame):
                            self.zone4_phase = 'sign'
                            print("Switching to sign detection in Zone 4")
                    elif self.zone4_phase == 'sign':
                        if self.detect_final_sign(frame):
                            self.publish_velocity(0, 0)
                            tt1.run()
                            print("Sign detected. Navigation complete!")
                            self.is_auto_navigation = False

            self.ui_interface.show_image(output_frame)

            # FPS and processing time indicator
            if not self.last_status_update_time or (rospy.get_time() - self.last_status_update_time) > 1:
                fps = round(self.processed_frame_count / 1)
                self.processed_frame_count = 0
                processing_time_ms = (time.time_ns() - process_start_time) / 1E6
                status_info = f'FPS: {fps} \nProcessing time: {processing_time_ms:.2f} ms'
                self.ui_interface.update_status(status_info)
                self.last_status_update_time = rospy.get_time()

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)

    def detect_navigation_line(self, frame, lower_color, upper_color):
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_frame, lower_color, upper_color)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv.erode(mask, kernel, iterations=1)
        mask = cv.dilate(mask, kernel, iterations=2)

        frame_height, frame_width = mask.shape
        scanline_y = int(0.75 * frame_height)
        line_indices = np.where(mask[scanline_y, :] > 0)[0]

        if len(line_indices) > 0:
            line_position = int(np.mean(line_indices))
            alignment_error = (line_position - frame_width // 2) / (frame_width // 2)
            forward_speed = 2.5 * (1 - abs(alignment_error))
            angular_speed = -20 * alignment_error
            self.publish_velocity(forward_speed, angular_speed)
            cv.circle(frame, (line_position, scanline_y), 5, (0, 255, 0), -1)
        else:
            self.publish_velocity(0, 0)
            print("No navigation line detected")
        self.ui_interface.show_image(mask, feed=1)

    def align_with_tunnel(self, frame):
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_tunnel = np.array([5, 70, 150])
        upper_tunnel = np.array([15, 255, 200])
        mask = cv.inRange(hsv_frame, lower_tunnel, upper_tunnel)

        frame_height, frame_width = mask.shape
        roi = mask[int(0.6 * frame_height):, :]
        moments = cv.moments(roi)

        if moments['m00'] > 0:
            center_x = int(moments['m10'] / moments['m00'])
            alignment_error = (center_x - frame_width // 2) / (frame_width // 2)
            forward_speed = 2.5 * (1 - abs(alignment_error))
            angular_speed = -15 * alignment_error
            self.publish_velocity(forward_speed, angular_speed)
            cv.circle(frame, (center_x, int(0.8 * frame_height)), 5, (0, 255, 0), -1)
        else:
            self.publish_velocity(0, 0)
            print("No tunnel path detected")
        self.ui_interface.show_image(mask, feed=1)

    def detect_transition_line(self, frame, debug=False):
        # Define Region of Interest (ROI) - bottom quarter of the frame
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_height, frame_width = hsv_frame.shape[:2]
        roi = hsv_frame[int(0.75 * frame_height):, :]  # Bottom 25% of the frame

        # Calculate the mean HSV value in the ROI
        mean_hue = int(np.mean(roi[:, :, 0]))  # Hue channel
        mean_sat = int(np.mean(roi[:, :, 1]))  # Saturation channel
        mean_val = int(np.mean(roi[:, :, 2]))  # Value channel

        # Dynamic thresholds with tolerance
        hue_tolerance = 15
        sat_tolerance = 50
        val_tolerance = 50

        lower_pink = np.array([
            max(mean_hue - hue_tolerance, 140),  # Minimum hue around pink
            max(mean_sat - sat_tolerance, 50),   # Minimum saturation
            max(mean_val - val_tolerance, 50)    # Minimum value
        ])
        upper_pink = np.array([
            min(mean_hue + hue_tolerance, 179),  # Maximum hue in HSV
            min(mean_sat + sat_tolerance, 255),  # Maximum saturation
            min(mean_val + val_tolerance, 255)   # Maximum value
        ])

        # Create mask and compute pink area
        mask = cv.inRange(hsv_frame, lower_pink, upper_pink)
        pink_area = np.sum(mask > 0)

        # Confidence scoring
        if pink_area > 1000:  # Threshold for detection
            self.zone_confidence_score += 1
        else:
            self.zone_confidence_score = max(0, self.zone_confidence_score - 1)

        if self.zone_confidence_score > 5:
            self.zone_confidence_score = 0
            print("Pink transition line detected!")
            return True

        return False


    def detect_hill_path(self, frame):
        frame_height, frame_width = frame.shape[:2]
        roi = frame[int(frame_height / 2):, :]

        gray_frame = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray_frame, 50, 150)

        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
        if lines is not None:
            slopes = [(y2 - y1) / (x2 - x1 + 1e-6) for x1, y1, x2, y2 in lines[:, 0]]
            avg_slope = np.mean(slopes)
            if abs(avg_slope) < 0.5:
                self.publish_velocity(3.0, 0)
            elif avg_slope > 0.5:
                self.publish_velocity(3.0, -10)
            elif avg_slope < -0.5:
                self.publish_velocity(3.0, 10)
        else:
            self.publish_velocity(2.0, 0)
        self.ui_interface.show_image(edges, feed=1)

    def publish_velocity(self, forward_speed, angular_speed):
        twist_message = Twist()
        twist_message.linear.x = forward_speed
        twist_message.angular.z = angular_speed
        self.velocity_publisher.publish(twist_message)

    def update_score_tracker(self, position, prediction='NA'):
        self.score_publisher.publish(String(data=f'{TEAM_NAME},{PWD},{position},{prediction}'))

Position1 = [5.5, 2.5, 0.2, 0, 0, math.radians(-90)]
Position2 = [0.5, 0, 0.2, 0, 0, math.radians(90)]
Position3 = [-3.88, 0.41, 0.2, 0, 0, math.radians(180)]
Position4 = [-4, -2.3, 0.2, 0, 0, math.radians(0)]

PWD = 'Smash'
TEAM_NAME = 'Smash'
