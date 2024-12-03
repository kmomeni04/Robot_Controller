#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import cv2
import pandas as pd
import os
from std_msgs.msg import String


class Playback:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('driving_playback', anonymous=True)


        # Publisher to cmd_vel
        self.cmd_pub = rospy.Publisher('B1/cmd_vel', Twist, queue_size=10)

        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)


        # Ensure the working directory is correct
        os.chdir('/home/fizzer/ros_ws/src/ws_Controller/data')


        # Load movement commands
        self.commands_df = pd.read_csv('movement_commands.csv')

        


    def run(self):
        rospy.sleep(1)  # Wait for publisher connections
        start_time = rospy.get_time()
        self.pub_score.publish(String(data="sherlock,detectiv,0,AAAA"))

        num_commands = len(self.commands_df)
        index = 0


        # Start with the first command
        current_command = self.commands_df.iloc[index]
        current_timestamp = current_command['timestamp'] - self.commands_df.iloc[0]['timestamp']  # Normalize timestamps
        linear_x = current_command['linear_x']
        angular_z = current_command['angular_z']


        # Initialize the next command's timestamp
        if index + 1 < num_commands:
            next_command = self.commands_df.iloc[index + 1]
            next_timestamp = next_command['timestamp'] - self.commands_df.iloc[0]['timestamp']
        else:
            next_timestamp = current_timestamp + 1.0  # Default duration for the last command


        rate = rospy.Rate(30)  # Publish at 10 Hz


        while not rospy.is_shutdown():
            current_time = rospy.get_time() - start_time


            # Check if it's time to move to the next command
            if current_time >= next_timestamp:
                index += 1
                if index >= num_commands:
                    break  # No more commands
                else:
                    # Update current command
                    current_command = self.commands_df.iloc[index]
                    current_timestamp = current_command['timestamp'] - self.commands_df.iloc[0]['timestamp']
                    linear_x = current_command['linear_x']
                    angular_z = current_command['angular_z']


                    # Update next command's timestamp
                    if index + 1 < num_commands:
                        next_command = self.commands_df.iloc[index + 1]
                        next_timestamp = next_command['timestamp'] - self.commands_df.iloc[0]['timestamp']
                    else:
                        next_timestamp = current_timestamp + 1.0  # Default duration for the last command


            # Publish the current command
            twist = Twist()
            twist.linear.x = linear_x
            twist.angular.z = angular_z
            self.cmd_pub.publish(twist)


            rospy.loginfo(f"Time: {current_time:.2f}s, Published Twist: linear.x={twist.linear.x}, angular.z={twist.angular.z}")

            rate.sleep()


        # Stop the robot after playback
        stop_twist = Twist()
        self.cmd_pub.publish(stop_twist)
        rospy.loginfo("Playback completed. Robot stopped.")
        self.pub_score.publish(String(data="sherlock,detectiv,-1,AAAA"))


if __name__ == '__main__':
    try:
        playback = Playback()
        playback.run()
    except rospy.ROSInterruptException:
        pass











