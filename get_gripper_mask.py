import rospy
import baxter_interface
from baxter_interface.limb import Limb
from baxter_pykdl import baxter_kinematics
import numpy as np

from std_msgs.msg import Header
from baxter_core_msgs.msg import DigitalIOState, EndEffectorState
from sensor_msgs.msg import Image, CameraInfo

from PIL import Image as PIL_Image

import sys, os, time, cPickle

def to_rgb(bgra_img):
    # change the quirky BGRA channels from ROS
    data = np.asarray(bgra_img.getdata())
    b, g, r = [data[:, i] for i in xrange(3)]
    rgb = np.asarray(zip(r, g, b), dtype=np.uint8)
    return PIL_Image.fromarray(rgb.reshape((bgra_img.height, bgra_img.width, 3)))

rospy.init_node('get_gripper_mask')
rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
rs.enable()

# arms
right = baxter_interface.Limb('right')
left = baxter_interface.Limb('left')
left_gripper = baxter_interface.Gripper('left')
left_gripper.calibrate()

# retrieve images
global current_image
pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=1)
def update_image(msg):
    global current_image
    current_image = PIL_Image.frombytes('RGBA', (msg.width, msg.height), msg.data)
    msg.data = current_image.tobytes()
    pub.publish(msg)
rospy.Subscriber('/cameras/left_hand_camera/image', Image, update_image)

print 'saving image in 5 seconds...'
rospy.sleep(5.)
gripper_image = to_rgb(current_image)
gripper_image.save('gripper_mask.png')
