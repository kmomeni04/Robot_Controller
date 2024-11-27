#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class LineFollower:
    def __init__(self):
        # Initialize the node
        rospy.init_node('line_follower_bot', anonymous=True)

        # Publisher for robot movement
        self.pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        # Publisher for score tracking
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Subscriber for the image topic
        self.image_sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback)

        # Initialize Twist message for movement commands
        self.cmd_move = Twist()

        # Start time for score tracking
        self.start_time = rospy.Time.now()

    def image_callback(self, msg):
        """
        Callback function for processing incoming image messages.
        """
        # Check if 240 seconds have elapsed
        current_time = rospy.Time.now()
        elapsed_time = (current_time - self.start_time).to_sec()
        
        if elapsed_time >= 240:
            rospy.loginfo("240 seconds elapsed - shutting down node")
            # Publish final stop command
            self.cmd_move.linear.x = 0
            self.cmd_move.angular.z = 0
            self.pub.publish(self.cmd_move)
            # Shutdown the node
            rospy.signal_shutdown("Time limit reached")
            return

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Get image dimensions
        height, width, _ = cv_image.shape
        
        # Define region of interest (bottom portion of image)
        roi = cv_image[int(height*0.6):height, 0:width]
        
        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Calculate centroid of white pixels
        moments = cv2.moments(binary)
        
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            
            # Calculate error from center
            error = cx - width/2
            
            # PID constants
            Kp = 0.005
            
            # Calculate angular velocity based on error
            angular_z = -Kp * error
            
            # Set movement commands
            self.cmd_move.linear.x = 0.2  # Constant forward speed
            self.cmd_move.angular.z = angular_z
            
            # Publish movement command
            self.pub.publish(self.cmd_move)
        else:
            # If line is lost, stop the robot
            self.cmd_move.linear.x = 0
            self.cmd_move.angular.z = 0
            self.pub.publish(self.cmd_move)

        # Publish the elapsed time to the score tracker
        elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
        self.pub_score.publish(String(data=f"Elapsed Time: {elapsed_time:.2f} seconds"))

    def run(self):
        """
        Main run loop for the line follower node.
        """
        self.pub_score.publish(String(data="sherlock,detective,0,AAAA"))
        rate = rospy.Rate(10)  # 10Hz
        
        try:
            while not rospy.is_shutdown():
                # Check if 240 seconds have elapsed
                current_time = rospy.Time.now()
                elapsed_time = (current_time - self.start_time).to_sec()
                
                if elapsed_time >= 240:
                    rospy.loginfo("240 seconds elapsed - shutting down node")
                    # Stop the robot
                    self.cmd_move.linear.x = 0
                    self.cmd_move.angular.z = 0
                    self.pub.publish(self.cmd_move)
                    # Shutdown the node
                    rospy.signal_shutdown("Time limit reached")
                    break
                    
                rate.sleep()
                
        except rospy.ROSInterruptException:
            pass
        
        finally:
            # Ensure robot stops when node shuts down
            self.cmd_move.linear.x = 0
            self.cmd_move.angular.z = 0
            self.pub_score.publish(String(data="sherlock,detective,-1,AAAA"))
            self.pub.publish(self.cmd_move)

if __name__ == '__main__':
    try:
        line_follower = LineFollower()
        line_follower.run()
    except rospy.ROSInterruptException:
        pass
