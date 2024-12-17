from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow
from python_qt_binding import loadUi
import sys
import cv2 as cv
from robot_controller import RobotController


class RobotGui(QMainWindow):
    image_signal = pyqtSignal(QPixmap, int)

    def __init__(self):
        super(RobotGui, self).__init__()
        loadUi("./gui.ui", self)

        self.pressed_keys = set()
        self.robot_controller = RobotController(self)

        # Connect UI buttons to their respective handlers
        self.tp1_btn.clicked.connect(self.on_tp1_clicked)
        self.tp2_btn.clicked.connect(self.on_tp2_clicked)
        self.tp3_btn.clicked.connect(self.on_tp3_clicked)
        self.tp4_btn.clicked.connect(self.on_tp4_clicked)

        self.start_btn.clicked.connect(self.on_start_clicked)
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.save_img_btn.clicked.connect(self.on_save_primary_image_clicked)
        self.save_img2_btn.clicked.connect(self.on_save_secondary_image_clicked)

        self.image_signal.connect(self.refresh_image_label)

        # Timer for GUI-safe image updates
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.update_image_feed)
        self.image_timer.start(100)  # Update every 100 ms

        # Initial status message
        self.set_status_message("Ready")

    # Button Event Handlers
    def on_tp1_clicked(self):
        try:
            self.robot_controller.tp1()
        except Exception as e:
            self.set_status_message(f"Error: {e}")

    def on_tp2_clicked(self):
        try:
            self.robot_controller.tp2()
        except Exception as e:
            self.set_status_message(f"Error: {e}")

    def on_tp3_clicked(self):
        try:
            self.robot_controller.tp3()
        except Exception as e:
            self.set_status_message(f"Error: {e}")

    def on_tp4_clicked(self):
        try:
            self.robot_controller.tp4()
        except Exception as e:
            self.set_status_message(f"Error: {e}")

    def on_start_clicked(self):
        try:
            self.robot_controller.start()
            self.set_status_message("Navigation started")
        except Exception as e:
            self.set_status_message(f"Error: {e}")

    def on_stop_clicked(self):
        try:
            self.robot_controller.stop()
            self.set_status_message("Navigation stopped")
        except Exception as e:
            self.set_status_message(f"Error: {e}")

    def on_save_primary_image_clicked(self):
        try:
            self.robot_controller.save_img()
            self.set_status_message("Primary image saved")
        except Exception as e:
            self.set_status_message(f"Error: {e}")

    def on_save_secondary_image_clicked(self):
        try:
            self.robot_controller.save_img2()
            self.set_status_message("Secondary image saved")
        except Exception as e:
            self.set_status_message(f"Error: {e}")

    # Key Event Handling
    def keyPressEvent(self, event):
        if event.key() not in self.pressed_keys:
            self.pressed_keys.add(event.key())
            self.handle_key_based_movement()

    def keyReleaseEvent(self, event):
        if event.key() in self.pressed_keys:
            self.pressed_keys.remove(event.key())
            self.handle_key_based_movement()

    def handle_key_based_movement(self):
        if not self.pressed_keys:
            self.robot_controller.move_robot(0, 0)  # Stop if no keys are pressed
            return

        vx, wz = 0, 0
        if Qt.Key_W in self.pressed_keys:
            vx += 2
        if Qt.Key_S in self.pressed_keys:
            vx -= 2
        if Qt.Key_A in self.pressed_keys:
            wz += 3
        if Qt.Key_D in self.pressed_keys:
            wz -= 3
        self.robot_controller.move_robot(vx, wz)

    # Image Handling
    def update_image_feed(self):
        """
        Periodically request the latest image frames from the robot controller
        and update the GUI.
        """
        try:
            frame = self.robot_controller.get_camera_feed()
            if frame is not None:
                self.display_image(frame)
        except Exception as e:
            self.set_status_message(f"Image Update Error: {e}")

    def display_image(self, cv_img, feed=0):
        try:
            if cv_img is None:
                return
            cv_img = cv.resize(cv_img, (400, 300))
            cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
            height, width, channel = cv_img.shape
            bytesPerLine = channel * width
            q_img = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            qt_img = QPixmap.fromImage(q_img)

            self.image_signal.emit(qt_img, feed)
        except Exception as e:
            print(f"Error displaying image: {e}")

    def refresh_image_label(self, qt_img, feed):
        if feed == 0:
            self.feed_0.setPixmap(qt_img)
        elif feed == 1:
            self.feed_1.setPixmap(qt_img)
        elif feed == 2:
            self.feed_2.setPixmap(qt_img)

    def set_status_message(self, status_str):
        self.status_label.setText(status_str)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RobotGui()
    window.show()
    sys.exit(app.exec_())
