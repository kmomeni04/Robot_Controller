#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# Initialize the ROS node
rospy.init_node('topic_publisher')

# Set the rate to 10 Hz
rate = rospy.Rate(10)

# Publishers
pub_cmd = rospy.Publisher('B1/cmd_vel', Twist, queue_size=1)
pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)
rospy.sleep(1)

# Create a Twist message for movement
move = Twist()
move.linear.x = 1  # Move forward with speed 1 m/s

# Publish a score tracker message
rospy.sleep(1)
pub_score.publish(String(data='sherlock,detective,0,AAAA'))

# Publish the movement command continuously for 5 seconds
start_time = rospy.Time.now()
while rospy.Time.now() - start_time < rospy.Duration(5):  # 5 seconds
    pub_cmd.publish(move)
    rate.sleep()
move.linear.x = 0  # Stop the robot
pub_cmd.publish(move)
pub_score.publish(String(data='sherlock,detective,-