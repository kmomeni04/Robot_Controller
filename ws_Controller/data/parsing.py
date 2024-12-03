import pandas as pd
import numpy as np
import os

os.chdir('/home/fizzer/ros_ws/src/ws_Controller/data')
commands_df = pd.read_csv('movement_commands.csv')

# Function to get command for a given frame number
def get_command_for_frame(frame_num, fps, commands_df):
    frame_time = frame_num / fps
    # Find the closest timestamp
    closest_idx = np.argmin(np.abs(commands_df['timestamp'] - frame_time))
    command = commands_df.iloc[closest_idx]
    return command['linear_x'], command['angular_z']

# Example: Associating commands with each frame in the video
import cv2

video_path = 'driving_recording.avi'
cap = cv2.VideoCapture(video_path)

frame_num = 0
fps = 20.0  # Ensure this matches the fps used during recording

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    linear_x, angular_z = get_command_for_frame(frame_num, fps, commands_df)
    print(f"Frame {frame_num}: linear.x = {linear_x}, angular.z = {angular_z}")

    # Process the frame as needed
    frame_num += 1

cap.release()


