import rospy
import baxter_interface
from baxter_interface.limb import Limb
from baxter_pykdl import baxter_kinematics
import numpy as np

from std_msgs.msg import Header
from baxter_core_msgs.msg import DigitalIOState, EndEffectorState
from sensor_msgs.msg import Image, CameraInfo
from ar_track_alvar_msgs.msg import AlvarMarkers
from image_geometry import PinholeCameraModel

from tf import TransformListener

from PIL import Image as PIL_Image
from PIL import ImageDraw, ImageFont

import sys, os, time, cPickle, json, argparse

def to_rgb(bgra_img):
    # change the quirky BGRA channels from ROS
    data = np.asarray(bgra_img.getdata())
    b, g, r = [data[:, i] for i in xrange(3)]
    rgb = np.asarray(zip(r, g, b), dtype=np.uint8)
    return PIL_Image.fromarray(rgb.reshape((bgra_img.height, bgra_img.width, 3)))

rospy.init_node('get_gripper_mask')
rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
rs.enable()

# change camera resolution
cc = baxter_interface.CameraController('right_hand_camera')
cc.resolution = (1280, 800)
msg = rospy.wait_for_message('/cameras/right_hand_camera/camera_info', CameraInfo)
camera = PinholeCameraModel()
camera.fromCameraInfo(msg)
font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf', size=32)

tl = TransformListener()

# retrieve images
global current_image
pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=1)
def update_image(msg):
    global current_image
    current_image = PIL_Image.frombytes('RGBA', (msg.width, msg.height), msg.data)
rospy.Subscriber('/cameras/right_hand_camera/image', Image, update_image)

global workspace
global markers
markers = {}
def update_markers(msg):
    global current_image, markers
    display_image = current_image.copy()
    if len(msg.markers) > 0:
        draw = ImageDraw.Draw(display_image)
        # draw markers
        for marker in msg.markers:
            marker.pose.header = marker.header
            transformed_pose = tl.transformPose('/right_hand_camera', marker.pose)
            transformed_xyz = (transformed_pose.pose.position.x, transformed_pose.pose.position.y, transformed_pose.pose.position.z)
            xyz_str = map(lambda s: '%g' % s, (marker.pose.pose.position.x, marker.pose.pose.position.y, marker.pose.pose.position.z))
            u, v = camera.project3dToPixel(transformed_xyz)
            draw.ellipse([(u-10, v-10), (u+10, v+10)], fill=(0, 255, 0), outline=(0, 255, 0))
            draw.text((u+10, v+10), str(', '.join(xyz_str)), font=font)
            # update markers
            markers[marker.id] = {
                'id': marker.id,
                'frame_id': '/base',
                'position': (marker.pose.pose.position.x, marker.pose.pose.position.y, marker.pose.pose.position.z),
            }
    # scale image and update display
    width = 1280 * 600 / 800
    display_image = display_image.resize((width, 600))
    display_msg = Image(
        header=Header(
            stamp=rospy.Time.now(),
            frame_id='base',
        ),
        width=width,
        height=600,
        step=width * 4,
        encoding='bgra8',
        is_bigendian=0,
        data=display_image.tobytes(),
    )
    pub.publish(display_msg)
rospy.Subscriber('/ar_pose_marker', AlvarMarkers, update_markers)

def save_workspace(msg):
    if msg.state == DigitalIOState.PRESSED:
        for i in markers:
            fn = 'marker.%i.json' % i
            with open(fn, 'wb') as f:
                json.dump(markers[i], f)
            print markers[i]
            print 'marker %i saved to %s' % (i, fn)
rospy.Subscriber('/robot/digital_io/right_upper_button/state', DigitalIOState, save_workspace)

def finish(msg):
    if msg.state == DigitalIOState.PRESSED:
        rospy.signal_shutdown('done')
rospy.Subscriber('/robot/digital_io/right_lower_button/state', DigitalIOState, finish)

print 'save markers to JSON when the large cuff button is pressed'
print 'exit when the small cuff button is pressed'

rospy.spin()
