#!/usr/bin/env python3


import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import pandas as pd
import time
import os


class DataCollector:
   def __init__(self):
       # Initialize ROS node
       rospy.init_node('data_collector', anonymous=True)


       # Parameters
       self.video_filename = rospy.get_param('~video_filename', 'driving_recording.avi')
       self.csv_filename = rospy.get_param('~csv_filename', 'movement_commands.csv')
       self.frame_width = rospy.get_param('~frame_width', 640)
       self.frame_height = rospy.get_param('~frame_height', 480)
       self.fps = rospy.get_param('~fps', 30.0)


       # Initialize CvBridge
       self.bridge = CvBridge()


       # Initialize Video Writear
       fourcc = cv2.VideoWriter_fourcc(*'XVID')
       self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, self.fps, (self.frame_width, self.frame_height))


       # Initialize DataFrame to store movement commands
       self.commands = []
       self.start_time = rospy.get_time()


       # Subscribe to image and cmd_vel topics
       self.image_sub = rospy.Subscriber('/B1/rrbot/camera_birdseye/image_raw_birdseye', Image, self.image_callback)
       self.cmd_vel_sub = rospy.Subscriber('B1/cmd_vel', Twist, self.cmd_vel_callback)


       rospy.loginfo("Data Collector Initialized.")
  
   def image_callback(self, data):
       try:
           # Convert ROS Image message to OpenCV image
           cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
       except CvBridgeError as e:
           rospy.logerr(f"CvBridge Error: {e}")
           return


       # Optionally, resize the image if needed
       if cv_image.shape[1] != self.frame_width or cv_image.shape[0] != self.frame_height:
           cv_image = cv2.resize(cv_image, (self.frame_width, self.frame_height))


       # Write frame to video
       self.video_writer.write(cv_image)


       # Display the frame (optional)
       cv2.imshow('Recording', cv_image)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           rospy.signal_shutdown("User pressed 'q'")


   def cmd_vel_callback(self, msg):
       # Capture the timestamp relative to the start of recording
       timestamp = rospy.get_time() - self.start_time
       linear_x = msg.linear.x
       angular_z = msg.angular.z


       # Append to the list
       self.commands.append({'timestamp': timestamp, 'linear_x': linear_x, 'angular_z': angular_z})


       rospy.logdebug(f"Time: {timestamp:.2f}s, linear.x: {linear_x}, angular.z: {angular_z}")


   def save_data(self):
       # Release the video writer
       self.video_writer.release()
       cv2.destroyAllWindows()


       # Convert commands list to DataFrame
       df = pd.DataFrame(self.commands)


       # Save to CSV
       df.to_csv(self.csv_filename, index=False)
       rospy.loginfo(f"Saved movement commands to {self.csv_filename}")
       rospy.loginfo(f"Saved video recording to {self.video_filename}")


   def run(self):
       try:
           rospy.spin()
       except KeyboardInterrupt:
           rospy.loginfo("Shutting down data collector.")
       finally:
           self.save_data()


if __name__ == '__main__':
   collector = DataCollector()
   collector.run()


