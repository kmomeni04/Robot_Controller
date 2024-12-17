#!/usr/bin/env python3
import queue
import threading
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from tensorflow.keras.models import load_model
import string
from PIL import ImageFont
import os
import sys
import traceback

# Initialize constants
# Initialize constants
CHARACTER_MODEL_PATH = "/home/fizzer/Downloads/353/character_recognition_model.h5"
LABELS = string.ascii_uppercase + string.digits
SCORE_TRACKER_TOPIC = "/score_tracker"
CLUEBOARD_PATH = "/home/fizzer/ros_ws/src/ws_controller/src/clueboards/"
IMAGE_FORMAT = "clueboard_{}.png"
temp_folder="/home/fizzer/ros_ws/src/ws_controller/src/temp"
class SignDetector:
    def __init__(self):
        #rospy.init_node("sign_detector", anonymous=True)
        # self.bridge = CvBridge()

        # Initialize the tag variable
        self.tag = rospy.get_param("~tag", "1")  # Default value is "1"

        # Load the character recognition model
        try:
            self.character_model = load_model(CHARACTER_MODEL_PATH)
            rospy.loginfo("Character recognition model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load character recognition model: {e}")
            sys.exit(1)

        # Queues for inter-thread communication
        self.image_queue = queue.Queue()
        self.display_queue = queue.Queue()

        # Set default HSV thresholds
        self.lower_blue = np.array([120, 0, 190])
        self.upper_blue = np.array([180, 255, 255])

        # Start the image processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Publishers and Subscribers
        self.score_pub = rospy.Publisher(SCORE_TRACKER_TOPIC, String, queue_size=10)
        # self.image_sub = rospy.Subscriber("/B1/camera2/image_raw", Image, self.image_callback)
        self.tag_sub = rospy.Subscriber("/set_tag", String, self.tag_callback)  # Subscriber to update the tag dynamically
    def tag_callback(self, msg):
        """Callback to update the tag dynamically."""
        self.tag = msg.data
        rospy.loginfo(f"Tag updated to: {self.tag}")

    # def image_callback(self, msg):
    #     try:
    #         # Convert ROS Image to OpenCV format
    #         try:
    #             cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    #         except CvBridgeError as e:
    #             rospy.logerr(f"CvBridge Error: {e}")
    #             return

    #         # Put the image into the processing queue
    #         self.image_queue.put(cv_image)

    #     except Exception as e:
    #         rospy.logerr(f"Error in image_callback: {e}")
    #         traceback.print_exc()

    def processing_loop(self):
        while not rospy.is_shutdown():
            try:
                # Get an image from the queue
                cv_image = self.image_queue.get(timeout=1)

                # Process the image
                self.process_image(cv_image)

            except queue.Empty:
                continue  # No image to process

            except Exception as e:
                rospy.logerr(f"Error in processing_loop: {e}")
                traceback.print_exc()

    def process_image(self, image_path, tag):
        rospy.loginfo(f"Processing image: {image_path} with tag: {tag}")
        frame = cv2.imread(image_path)
        if frame is None:
            rospy.logerr(f"Failed to load image: {image_path}")
            return
        if tag == 3 or tag==6:
            self.lower_blue = np.array([115, 0, 188])
            self.upper_blue = np.array([255, 255, 255])
        else:
            self.lower_blue = np.array([100, 125, 60])
            self.upper_blue = np.array([145, 255, 255])
        # Process the image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.01 * peri, True)

            if len(approx) == 4:
                rospy.loginfo("Found quadrilateral contour.")
                # Get the corner points and order them
                pts_src = np.array([point[0] for point in approx], dtype='float32')
                pts_src = self.order_points(pts_src)

                # Draw the contour on the original image for visualization
                # cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                self.display_queue.put(("Contour on Original", frame, False))

                # Define the destination points for homography (desired size)
                w, h = 600, 400  # Adjust based on your needs
                pts_dst = np.array([
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1]], dtype='float32')

                # Compute the homography matrix
                matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

                # Warp the image
                warped = cv2.warpPerspective(frame, matrix, (w, h))
                print(f"Warped image shape: {warped.shape}, dtype: {warped.dtype}")

                # Display the warped image before cropping
                self.display_queue.put(("Warped Image Before Cropping", warped, False))

                # Convert the warped image to HSV to detect the blue outline
                warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
                warped_mask = cv2.inRange(warped_hsv, self.lower_blue, self.upper_blue)

                # Invert the mask to focus on non-blue areas
                non_blue_mask = cv2.bitwise_not(warped_mask)
                # cv2.imshow("Non-Blue Mask", non_blue_mask)
                # cv2.waitKey(0)

                # Find contours in the non-blue mask
                contours_info = cv2.findContours(non_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

                if contours:
                    # Find the bounding rectangle of the largest non-blue contour
                    largest_non_blue_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_non_blue_contour)
                    # cv2.imshow("Non-Blue Mask", cv2.boundingRect(largest_non_blue_contour))
                    # cv2.waitKey(0)

                    # Crop the warped image dynamically
                    warped_cropped = warped[y:y + h, x:x + w]
                else:
                    rospy.loginfo("No non-blue region found, using full warped image.")
                    warped_cropped = warped.copy()

                # Display the dynamically cropped warped image
                self.display_queue.put(("Dynamically Cropped Warped Image", warped_cropped, False))

                # Convert warped image to grayscale for processing
                warped_gray = cv2.cvtColor(warped_cropped, cv2.COLOR_BGR2GRAY)

                # Save images for debugging
                cv2.imwrite(os.path.join(temp_folder, "warped_image.png"), warped)
                cv2.imwrite(os.path.join(temp_folder, "warped_cropped.png"), warped_cropped)
                cv2.imwrite(os.path.join(temp_folder, "warped_cropped_gray.png"), warped_gray)


                # Process the text regions separately
                clue_type, clue_value = self.process_text_region(warped_gray,tag)

                rospy.loginfo(f"Recognized clue type: {clue_type}, clue value: {clue_value}")
                # Publish the recognized clue type and clue value
                self.score_pub.publish(String(f"smash,smash,{tag},{clue_value}"))

                # Log instead of shutting down
                rospy.loginfo("Confident guess made. Continuing to the next image.")
            else:
                rospy.loginfo("No quadrilateral contour found.")
        else:
            rospy.loginfo("No contours found.")


    import os

    def process_text_region(self, image,tag):
        """
        Process the detected region of interest to extract and recognize clue type and clue value.
        Save the regions as images in the specified folder.
        """
        # Create the temp folder if it doesn't exist
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        h, w = image.shape

        # Extract the top-right region for clue type
        clue_type_region = image[:h // 3, (2 * w) // 5:]  # Top third and right two-fifths
        clue_type_path = os.path.join(temp_folder, "clue_type_region.png")
        cv2.imwrite(clue_type_path, clue_type_region)  # Save the clue type region
        rospy.loginfo(f"Clue type region saved to {clue_type_path}")

        # Extract the bottom region for clue value
        clue_value_region = image[h // 2:, :]  # Bottom half
        clue_value_path = os.path.join(temp_folder, "clue_value_region.png")
        cv2.imwrite(clue_value_path, clue_value_region)  # Save the clue value region
        rospy.loginfo(f"Clue value region saved to {clue_value_path}")

        # Recognize text in both regions
        clue_type_text = self.recognize_text(clue_type_region,tag=tag)
        clue_value_text = self.recognize_text(clue_value_region,tag=tag)

        return clue_type_text, clue_value_text


    def recognize_text(self, region, tag, expected_label=""):
        """
        Enhanced text recognition method with bounding box segmentation similar to the testing script.
        Args:
            region (numpy.ndarray): Grayscale image of the region to recognize text from.
            expected_label (str): The expected text in the region (for reference only).
        Returns:
            str: The recognized text.
        """
        try:
            # Preprocess the region
            binary = self.preprocess_image(region,tag)
            binaryfile = os.path.join(temp_folder, f"region{region}.png")
            cv2.imwrite(binaryfile, binary)


            # Use font metrics or scale dynamically based on the expected label
            if expected_label:
                FONT_PATH = "node/UbuntuMono-R.ttf"
                monospace_font = ImageFont.truetype(FONT_PATH, 90)
                char_width, _ = monospace_font.getsize("A")
                total_text_width, _ = monospace_font.getsize(expected_label)

                # Calculate scaling factor based on the text region
                vertical_sum = np.sum(binary, axis=0)
                non_zero_columns = np.where(vertical_sum > 0)[0]

                if non_zero_columns.size > 0:
                    text_start = non_zero_columns[0]
                    text_end = non_zero_columns[-1]
                    text_width = text_end - text_start + 1
                else:
                    text_start = 0
                    text_end = binary.shape[1] - 1
                    text_width = binary.shape[1]

                scale_factor = text_width / total_text_width
                adjusted_char_width = int(char_width * scale_factor)

                # Build bounding boxes
                bounding_boxes = []
                x_offset = text_start
                for _ in range(len(expected_label)):
                    x = x_offset
                    w = adjusted_char_width
                    if x + w > text_end:
                        w = text_end - x + 1
                    bounding_boxes.append((x, 0, w, binary.shape[0]))
                    x_offset += w

            else:
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bounding_boxes = [
                    cv2.boundingRect(cnt) for cnt in contours
                    if cv2.boundingRect(cnt)[2] >= 10 and cv2.boundingRect(cnt)[3] >= 10  # Filter by width and height
                ]
                bounding_boxes.sort(key=lambda b: b[0])  # Sort left-to-right

            # Recognize characters from each bounding box
            recognized_text = ""
            counter =0
            for i, (x, y, w, h) in enumerate(bounding_boxes):
                char_img = binary[y:y+h, x:x+w]

                # Add padding of 8 pixels on top and bottom, and 2 pixels on the sides
                top_padding = 48
                bottom_padding = 42
                side_padding = 5
                char_img_padded = cv2.copyMakeBorder(
                    char_img,
                    top_padding,  # Top padding
                    bottom_padding,  # Bottom padding
                    side_padding,        # Left padding
                    side_padding,        # Right padding
                    cv2.BORDER_CONSTANT,
                    value=0              # Black padding
                )


                # Resize to the model's input size
                char_img_resized = cv2.resize(char_img_padded, (28, 28))
                char_filename = os.path.join(temp_folder, f"char_{counter}.png")
                counter+=1
                cv2.imwrite(char_filename, char_img_resized)
                rospy.loginfo(f"Saved character {i} image to {char_filename}")

                # self.display_queue.put(("Character Image", char_img_resized, False))

                # Normalize and prepare for model input
                char_img_normalized = char_img_resized.astype('float32') / 255.0
                char_img_expanded = np.expand_dims(char_img_normalized, axis=-1)  # Add channel dimension
                char_img_expanded = np.expand_dims(char_img_expanded, axis=0)     # Add batch dimension

                # Predict the character
                prediction = self.character_model.predict(char_img_expanded)
                predicted_char = LABELS[np.argmax(prediction)]

                # Confidence threshold
                confidence = np.max(prediction)
                if confidence > 0.7:  # Adjust confidence threshold as needed
                    recognized_text += predicted_char
                    rospy.loginfo(f"Predicted character {i}: {predicted_char} with confidence {confidence:.2f}")
            return recognized_text

        except Exception as e:
            rospy.logerr(f"Error in recognize_text: {e}")
            traceback.print_exc()
            return ""
    @staticmethod
    def preprocess_image(image,tag):
        """
        Preprocess the input image to clean noise and prepare for character segmentation.
        """
        # Apply Gaussian Blur to reduce noise and improve thresholding results
        blurred = cv2.GaussianBlur(image, (7, 7), 0)

        if tag == 3 or tag==6:
            # Adaptive thresholding for uneven lighting conditions

            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 9, 3  # Using THRESH_BINARY_INV as in training code
            )
        else:
            # Adaptive thresholding for uneven lighting conditions
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 9, 3  # Using THRESH_BINARY_INV as in training code
            )

        # Morphological operations to clean noise and connect broken parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Suppress noise near the edges
        binary[:5, :] = 0  # Top edge
        binary[-5:, :] = 0  # Bottom edge
        binary[:, :5] = 0  # Left edge
        binary[:, -5:] = 0  # Right edge

        return binary

    @staticmethod
    def prepare_for_model(char_img):
        """
        Prepares a character image for input into the CNN model.
        """
        # Resize to model input size
        char_img = cv2.resize(char_img, (28, 28))

        # Normalize the image
        char_img = char_img.astype('float32') / 255.0  # Normalize to [0, 1]

        # Expand dimensions to match model input
        char_img = np.expand_dims(char_img, axis=-1)  # Add channel dimension
        char_img = np.expand_dims(char_img, axis=0)   # Add batch dimension

        return char_img

    @staticmethod
    def order_points(pts):
        """
        Orders the points in the order: top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype="float32")

        # Sum and difference of points
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        # Assign points based on sums and differences
        rect[0] = pts[np.argmin(s)]        # Top-left
        rect[2] = pts[np.argmax(s)]        # Bottom-right
        rect[1] = pts[np.argmin(diff)]     # Top-right
        rect[3] = pts[np.argmax(diff)]     # Bottom-left

        return rect

    def run(self):
        detector = self

        # # Create OpenCV windows
        # cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Blue Mask", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Character Image", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Clue Type Region", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Clue Value Region", cv2.WINDOW_NORMAL)

        try:
            # Loop through clueboard images
            for tag in range(1, 9):  # clueboard_1.png to clueboard_8.png
                image_path = os.path.join(CLUEBOARD_PATH, IMAGE_FORMAT.format(tag))
                detector.process_image(image_path, tag)
                rospy.sleep(1)  # Add a small delay between processing
        except rospy.ROSInterruptException:
            rospy.loginfo("Shutting down sign detector.")
        except Exception as e:
            rospy.logerr(f"An error occurred: {e}")
            traceback.print_exc()




