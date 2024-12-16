#! /usr/bin/env python3
import math
import rospy
import numpy as np
import datetime
import time
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import tf.transformations as tft
import cv2 as cv

# Credentials
TEAM_NAME = 'Smash'
PASSWORD = 'Smash'

# Teleportation Points (TP)
NAV_POINTS = {
    1: [5.5, 2.5, 0.2, 0, 0, math.radians(-90)],
    2: [0.5, 0, 0.2, 0, 0, math.radians(90)],
    3: [-3.88, 0.41, 0.2, 0, 0, math.radians(180)],
    4: [-4, -2.3, 0.2, 0, 0, math.radians(0)]
}

def clamp(val, low, high):
    return max(min(val, high), low)

class AutonomousNavigator:
    def __init__(self, gui_app):
        # GUI integration
        self.gui = gui_app

        # ROS initialization
        rospy.init_node('autonomous_navigation', anonymous=True)

        # Publishers
        self.pub_velocity = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.pub_tracker = rospy.Publisher('/score_tracker', String, queue_size=1)

        # Bridge setup for image conversion
        self.bridge = CvBridge()
        rospy.Subscriber('/B1/camera1/image_raw', Image, self.primary_cam_cb)

        self.secondary_subscriber = None
        self.last_secondary_unsub = rospy.Time.now()

        # Flags and states
        self.is_autonomous = False
        self.capture_primary_img = False
        self.capture_secondary_img = False
        self.enable_scoring = True
        self.begin_time = 0
        
        self.frame_counter = 0
        self.last_highres_frame = None

        # Yoda detection template
        self.yoda_template = cv.imread('templates/yoda_face.png', cv.IMREAD_GRAYSCALE)
        if self.yoda_template is None:
            rospy.logwarn("Yoda template not found.")
            self.yoda_template = np.zeros((50,50), dtype=np.uint8)
        self.template_h, self.template_w = self.yoda_template.shape
        self.yoda_match_threshold = 0.7

        self.CLUE_PIXEL_THRESHOLD = 1000
        self.reset_mission_state()

    def reset_mission_state(self):
        # Mission parameters
        self.current_zone = 0
        self.last_zone_change = rospy.get_time()
        self.mission_sequence = [
            {'zone': 1, 'action': 'line_following', 'params': {'speed': 2.0}},
            {'zone': 2, 'action': 'clue_search', 'params': {'clue_type': 'symbol'}},
            {'zone': 3, 'action': 'obstacle_course', 'params': {'maneuvers': ['left', 'right', 'swerve']}},
            {'zone': 4, 'action': 'final_approach', 'params': {'speed': 1.5}}
        ]
        self.current_mission_step = 0
        self.mission_active = True
        self.mission_start_time = rospy.get_time()

        self.flags = {'emergency_stop': False, 'manual_override': False}
        self.DETECTION_THRESHOLDS = {'clue': 1200, 'obstacle': 800}

        self.collected_clues = []
        self.obstacle_avoidance_log = []
        self.navigation_path = []

        # Additional variables
        self.energy_level = 100
        self.communication_status = 'connected'
        self.error_log = []

        # Zone-specific variables
        self.z3_mode = 'hill'
        self.z3_hill_timer = None
        self.z3_hill_paused = False
        self.z3_hill_actions = []
        self.z3_seen_pink = False

        self.z4_mode = 'tunnel'
        self.z4_tunnel_start = rospy.get_time()
        self.z4_last_left_line = 0
        self.current_clue_index = 0

    def start_autonomy(self):
        self.begin_time = rospy.get_time()
        print('Autonomous navigation initiated.')
        self.is_autonomous = True
        if self.enable_scoring:
            self.update_tracker(0)

    def halt_autonomy(self):
        if self.is_autonomous:
            print('Autonomy halted by manual intervention.')
            self.drive(0, 0)
            self.is_autonomous = False
            if self.enable_scoring:
                self.update_tracker(-1)

    def request_img_capture(self):
        self.capture_primary_img = True

    def request_secondary_img_capture(self):
        self.capture_secondary_img = True

    def secondary_cam_cb(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        filename = f'clueboards/clueboard_{self.current_clue_index - 1}.png'
        cv.imwrite(filename, frame)
        print(f'Saved secondary camera clueboard image: {filename}')

        self.capture_secondary_img = False
        if self.secondary_subscriber is not None:
            self.secondary_subscriber.unregister()
            self.secondary_subscriber = None
            self.last_secondary_unsub = rospy.Time.now()

    def primary_cam_cb(self, img_msg):
        try:
            start_proc = time.time_ns()
            frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            self.frame_counter += 1

            if self.capture_primary_img:
                filename = f'images/img_{datetime.datetime.now().strftime("%m%d_%H:%M:%S")}.png'
                cv.imwrite(filename, frame)
                print(f'Primary camera image saved: {filename}')
                self.capture_primary_img = False

            if self.is_autonomous:
                self.detect_clueboards(frame)
                # Zone-based logic
                if self.current_zone in [1, 2]:
                    self.zone_2_navigation(frame)
                elif self.current_zone == 3:
                    self.zone_3_navigation(frame)
                elif self.current_zone == 4:
                    self.zone_4_navigation(frame)

            self.gui.show_image(frame)
            elapsed_ms = (time.time_ns() - start_proc) / 1e6
            status = f'Processing Time: {elapsed_ms:.2f}ms'
            self.gui.update_status(status)

        except CvBridgeError as e:
            rospy.logerr("Bridge Error: %s", e)

    def detect_clueboards(self, frame):
        if self.current_clue_index in [3, 6]:
            lower_color = np.array([198, 0, 0])
            upper_color = np.array([228, 118, 106])
        else:
            lower_color = np.array([100, 0, 0])
            upper_color = np.array([150, 80, 80])

        mask = self.create_hsv_mask(frame, lower_color, upper_color, erosion_iterations=2, dilation_iterations=3)
        mask = self.retain_top_contours(mask, top_k=1)

        if np.sum(mask) > self.CLUE_PIXEL_THRESHOLD:
            self.current_clue_index += 1
            if self.secondary_subscriber is None:
                self.secondary_subscriber = rospy.Subscriber('/B1/camera2/image_raw', Image, self.secondary_cam_cb)

    def zone_2_navigation(self, frame):
        H, W = frame.shape[:2]
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        pink_lower_hsv = np.array([140, 100, 100])
        pink_upper_hsv = np.array([170, 255, 255])
        pink_mask = cv.inRange(hsv, pink_lower_hsv, pink_upper_hsv) // 255
        pink_mask = cv.erode(pink_mask, None, iterations=1)
        pink_mask = cv.dilate(pink_mask, None, iterations=2)
        pink_detected = (np.sum(pink_mask) / pink_mask.size) > 0.02
        self.gui.show_image(pink_mask * 255, feed=1)

        if pink_detected:
            self.current_zone += 1
            self.last_zone_change = rospy.get_time()
            print(f'Zone transition: now in zone {self.current_zone}')

    def zone_3_navigation(self, frame):
        yoda_detected, yoda_mask = self.new_check_yoda_presence(frame)
        if yoda_detected:
            print('Yoda detected in Zone 3!')
            self.handle_yoda_detection(yoda_mask)
            return

        if self.z3_mode == 'hill':
            self.navigate_hill()
        elif self.z3_mode == 'line':
            self.follow_purple_line(frame)
        elif self.z3_mode == 'tunnel':
            self.navigate_tunnel(frame)

    def handle_yoda_detection(self, yoda_mask):
        self.gui.show_image(yoda_mask * 255, feed=1)
        self.drive(0, 0)
        if self.z3_mode == 'hill':
            self.pause_hill_navigation()

    def pause_hill_navigation(self):
        if not self.z3_hill_paused:
            if self.z3_hill_timer:
                elapsed = rospy.get_time() - self.z3_hill_timer
                self.z3_hill_actions[0][2] -= elapsed
                self.z3_hill_timer = None
        self.z3_hill_paused = True

    def navigate_hill(self):
        if self.z3_hill_paused:
            return
        if not self.z3_hill_actions:
            print('Hill navigation completed. Switching to line mode.')
            self.z3_mode = 'line'
        else:
            if not self.z3_hill_timer:
                task = self.z3_hill_actions[0]
                self.drive(task[0], task[1])
                self.z3_hill_timer = rospy.get_time()
            elif rospy.get_time() - self.z3_hill_timer > self.z3_hill_actions[0][2] * 1.1:
                self.z3_hill_actions.pop(0)
                self.z3_hill_timer = None

    def navigate_tunnel(self, frame):
        """
        Navigate through the tunnel by detecting a specified color region to align the robot.
        """
        # Example HSV values and logic:
        color_lower = np.array([5, 50, 50])
        color_upper = np.array([30, 255, 150])
        tunnel_mask = cv.inRange(frame, color_lower, color_upper) // 255
        self.gui.show_image(tunnel_mask * 255, feed=1)

        mean_x = self.get_mean_contour_x(tunnel_mask)
        alignment_error = self.get_alignment_error(mean_x, tunnel_mask.shape[1])
        coverage = np.sum(tunnel_mask) / (tunnel_mask.size)

        if abs(alignment_error) > 0.18 or coverage < 0.12:
            self.drive(0, 7)
        else:
            self.drive(0, 0)
            self.current_zone += 1
            self.last_zone_change = rospy.get_time()

        yoda_present, yoda_mask = self.new_check_yoda_presence(frame)
        if yoda_present:
            print('Yoda encountered!')
            if self.z3_mode == 'hill':
                self.pause_hill_navigation()
            self.drive(0, 0)

        elif self.z3_mode == 'hill':
            # Resume hill navigation if paused and actions remain
            self.z3_hill_paused = False
            if not self.z3_hill_actions:
                print('Hill actions completed. Transitioning to line detection.')
                self.drive(0, 0)
                self.z3_mode = 'line'
            else:
                self.navigate_hill()
        elif self.z3_mode == 'line':
            self.follow_purple_line(frame)
        # Additional conditions for other modes if needed

    def zone_4_navigation(self, frame):
        H, W = frame.shape[:2]
        if self.z4_mode == 'tunnel':
            green_lower = np.array([60, 118, 102])
            green_upper = np.array([108, 200, 162])
            road_mask = cv.inRange(frame, green_lower, green_upper) // 255
            road_mask = cv.dilate(road_mask, None, iterations=2)
            road_mask = self.retain_top_contours(road_mask, top_k=1)
            self.gui.show_image(road_mask * 255, feed=1)

            mean_x = self.get_mean_contour_x(road_mask)
            error = self.get_alignment_error(mean_x, W)
            self.drive(3 * (1 - abs(error)), -15 * error)

            if rospy.get_time() - self.z4_tunnel_start > 2:
                self.z4_mode = 'mountain'

        elif self.z4_mode == 'mountain':
            lower_b = int(0.9 * (np.mean(frame[int(-0.2 * H):, :, 0]) - 75)) + 100
            lower_b = clamp(lower_b, 0, 255)
            line_lower = np.array([lower_b, 152, 0])
            line_upper = np.array([188, 253, 255])

            line_mask = cv.inRange(frame, line_lower, line_upper) // 255
            line_mask = cv.erode(line_mask, None, iterations=2)
            line_mask = cv.dilate(line_mask, None, iterations=2)
            line_mask = self.remove_small_areas(line_mask, 500)
            self.gui.show_image(line_mask * 255, feed=1)

            left = self.get_first_line_position(line_mask, int(-0.15 * H))
            if left > 0.35 * W:
                if self.z4_last_left_line == 0:
                    left = 0
                else:
                    left = self.z4_last_left_line
            self.z4_last_left_line = left
            loc = int(left + (0.36 * W))
            error = self.get_alignment_error(loc, W)
            vx = 3.5 * (1 - abs(error))
            wz = -20 * error
            self.drive(vx, wz)

            # Check for final sign
            sign_mask = cv.inRange(frame, np.array([0,0,0]), np.array([125,5,5])) // 255
            if np.sum(sign_mask) > 200:
                self.z4_mode = 'sign'

        elif self.z4_mode == 'sign':
            sign_mask = cv.inRange(frame, np.array([0,0,0]), np.array([125,5,5])) // 255
            self.gui.show_image(sign_mask*255, feed=1)

            mean_x = self.get_mean_contour_x(sign_mask)
            error = self.get_alignment_error(mean_x, W)
            self.drive(5*(1 - abs(error)), -20*error)

            if np.sum(sign_mask) > 3000:
                self.drive(0,0)
                print('Mission complete!')
                self.is_autonomous = False

    def new_check_yoda_presence(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(gray, self.yoda_template, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(res)
        mask = np.zeros_like(gray)
        if max_val >= self.yoda_match_threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + self.template_w, top_left[1] + self.template_h)
            cv.rectangle(mask, top_left, bottom_right, 255, -1)
            return True, mask
        return False, mask

    def follow_purple_line(self, frame):
        lower_hue = np.array([130, 50, 50])
        upper_hue = np.array([150, 255, 255])
        mask = self.create_hsv_mask(frame, lower_hue, upper_hue)
        self.gui.show_image(mask * 255, feed=1)

        edges = cv.Canny(mask, 50, 150, apertureSize=3)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=10)
        if lines is not None:
            angles = [math.atan2(y2 - y1, x2 - x1) for x1,y1,x2,y2 in lines[:,0]]
            mean_angle = np.mean(angles)

            angle_error = mean_angle
            angular_speed = -10 * angle_error
            linear_speed = 2.0 * (1 - abs(angle_error))

            total_line_pixels = np.sum(mask)
            if not self.z3_seen_pink and total_line_pixels > 6000:
                self.z3_seen_pink = True
            if self.z3_seen_pink and total_line_pixels < 5000:
                self.drive(0, 0)
                print('Purple line ended. Aligning for tunnel.')
                self.z3_mode = 'tunnel'
            else:
                self.drive(linear_speed, angular_speed)
        else:
            self.drive(0, 5)

    def update_tracker(self, location, prediction='NA'):
        msg = f'{TEAM_NAME},{PASSWORD},{location},{prediction}'
        self.pub_tracker.publish(String(data=msg))

    def drive(self, linear_x, angular_z):
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.pub_velocity.publish(cmd)

    def teleport_to(self, point_id):
        if point_id not in NAV_POINTS:
            print("Invalid teleportation point.")
            return
        self.drive(0,0)
        pose = NAV_POINTS[point_id]
        state_msg = ModelState()
        state_msg.model_name = 'B1'
        state_msg.pose.position.x = pose[0]
        state_msg.pose.position.y = pose[1]
        state_msg.pose.position.z = pose[2]
        q = tft.quaternion_from_euler(pose[3], pose[4], pose[5])
        state_msg.pose.orientation.x = q[0]
        state_msg.pose.orientation.y = q[1]
        state_msg.pose.orientation.z = q[2]
        state_msg.pose.orientation.w = q[3]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state_srv(state_msg)
        except rospy.ServiceException:
            print("Teleportation service call failed.")

        self.reset_mission_state()
        self.current_zone = point_id

    def create_hsv_mask(self, frame, lower_hsv, upper_hsv, erosion_iterations=1, dilation_iterations=2):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        mask = cv.erode(mask, None, iterations=erosion_iterations)
        mask = cv.dilate(mask, None, iterations=dilation_iterations)
        return mask // 255

    def retain_top_contours(self, mask, top_k=1):
        contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > top_k:
            areas = [cv.contourArea(c) for c in contours]
            largest_indices = np.argsort(areas)[::-1][:top_k]
            for i, c in enumerate(contours):
                if i not in largest_indices:
                    cv.drawContours(mask, [c], -1, 0, thickness=-1)
        return mask

    def remove_small_areas(self, mask, min_area):
        contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            if w*h < min_area:
                cv.drawContours(mask, [c], -1, 0, -1)
        return mask

    def get_mean_contour_x(self, mask):
        points = np.argwhere(mask == 1)
        if len(points) > 0:
            return int(np.mean(points[:, 1]))
        else:
            return mask.shape[1] // 2

    def get_first_line_position(self, mask, row_offset):
        H = mask.shape[0]
        target_row = H + row_offset
        if target_row < 0: 
            target_row = 0
        try:
            return np.where(mask[target_row, :] == 1)[0][0]
        except IndexError:
            return 0

    def get_alignment_error(self, mean_x, width):
        return (mean_x - width/2) / (width/2)
