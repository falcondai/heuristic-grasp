import rospy
import baxter_interface
from baxter_interface.limb import Limb
from baxter_pykdl import baxter_kinematics
import numpy as np
import tqdm

from std_msgs.msg import Header
from baxter_core_msgs.msg import DigitalIOState, EndEffectorState
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
from geometry_msgs.msg import PointStamped
from tf import TransformListener

from PIL import Image as PIL_Image
from PIL import ImageDraw as PIL_ImageDraw
from PIL import ImageFont
# import tensorflow as tf
from scipy import ndimage

import sys, os, time, cPickle

from motion_routine import *
from sklearn import cluster

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *

def to_rgb(bgra_img):
    # change the quirky BGRA channels from ROS
    data = np.asarray(bgra_img.getdata())
    b, g, r = [data[:, i] for i in xrange(3)]
    rgb = np.asarray(zip(r, g, b), dtype=np.uint8)
    return PIL_Image.fromarray(rgb.reshape((bgra_img.height, bgra_img.width, 3)))

def localize_cm_method(image, gripper_mask, percentile=1):
    gray = image.convert('L')
    gray_img = np.asarray(gray.getdata()).reshape((400, 640))
    m = np.percentile(gray_img[gripper_mask].flatten(), percentile)
    cv, cu = ndimage.center_of_mass((gray_img < m) * gripper_mask)
    return cu, cv

def localize_cluster_method(image, gripper_mask, model, percentile=2):
    gray = image.convert('L')
    # gray.save('gray.png')
    gray_img = np.asarray(gray.getdata()).reshape((400, 640))
    m = np.percentile(gray_img[gripper_mask].flatten(), percentile)
    imshow((gray_img < m) * gripper_mask)
    savefig('masked.png')
    xs = np.argwhere((gray_img < m) * gripper_mask)
    # scatter(xs[:, 0], xs[:, 1])
    # savefig('pre-cluster.png')
    ys = model.fit_predict(xs)
    n_clusters = ys.max()
    print 'n_clusters', n_clusters
    best_cu, best_cv = 0, 0
    debug_image = image.copy()
    draw = PIL_ImageDraw.Draw(debug_image)
    for i in xrange(n_clusters):
        cv, cu = xs[ys==i].mean(axis=0)
        if (cu - 320)**2 + (cv - 200)**2 < (best_cu - 320)**2 + (best_cv - 200)**2:
            best_cu, best_cv = cu, cv
        draw.ellipse([(cu-10, cv-10), (cu+10, cv+10)], outline=(0, 255, 0))
        draw.text((cu, cv), str(i))
    debug_image.save('clusters.png')
    return best_cu, best_cv


rospy.init_node('heuristic_grasp')
rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
rs.enable()

# task setup
n_attempts = int(sys.argv[1])
execution_error_threshold = 0.04
# z_initial = -0.05
z_initial = 0.
z_table = -0.165
grasp_force_threshold = 20.
# camera_pose = {'position': Limb.Point(x=0.7715770154184426, y=-0.3133978759217212, z=0.4350424834433179), 'orientation': Limb.Quaternion(x=-0.5303342084664359, y=0.6182759231154774, z=0.34040073425704037, w=0.469689099664077)}
# XYZ -> Left Anterior Superior
workspace_low = np.asarray([0.62, 0.62])
workspace_high = np.asarray([0.72, 0.72])

# set up camera
msg = rospy.wait_for_message('/cameras/left_hand_camera/camera_info', CameraInfo)
camera = PinholeCameraModel()
camera.fromCameraInfo(msg)

# arms
# right = baxter_interface.Limb('right')
left = baxter_interface.Limb('left')
left_gripper = baxter_interface.Gripper('left')
left_gripper.calibrate()

# transform listener
tl = TransformListener()

# retrieve images
global current_image, display_image, current_image_stamp
font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf', size=22)
pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=1)

# load gripper mask
gripper_mask_threshold = 30
gripper_mask = np.asarray(PIL_Image.open('gripper_mask.png').convert('L').getdata())
# print gripper_mask.max(), gripper_mask.min(), gripper_mask.mean()
gripper_mask = (gripper_mask > gripper_mask_threshold).reshape((400, 640))
gripper_mask = ndimage.binary_erosion(gripper_mask, iterations=8, border_value=1)
imshow(gripper_mask)
savefig('mask.png')
# gripper_mask = np.ones((400, 640))
# gripper_mask[:200, :] = 0

def update_image(msg):
    global current_image, current_image_stamp
    current_image = PIL_Image.frombytes('RGBA', (msg.width, msg.height), msg.data)
    current_image_stamp = msg.header.stamp
rospy.Subscriber('/cameras/left_hand_camera/image', Image, update_image)

global force
def update_gripper_state(msg):
    global force
    force = msg.force
rospy.Subscriber('/robot/end_effector/left_gripper/state', EndEffectorState, update_gripper_state)

# joint limits
# joint_limits = {
#     'left_w0': [0.28838838812245776, -1.8223691760078715],
#     'left_w1': [1.5826846779007757, 0.35051461003181705],
#     'left_w2': [3.056456719861687, -0.062126221909359257],
#     'left_e0': [0.6289321230330196, -0.2826359601678875],
#     'left_e1': [1.5512380717491248, 0.35089810522878839],
#     'left_s0': [-0.21015536794030168, -1.1075341288532687],
#     'left_s1': [-0.27458256103148904, -0.5],
# }
# joint_names = joint_limits.keys()
# joint_low = [a[1] for a in joint_limits.itervalues()]
# joint_high = [a[0] for a in joint_limits.itervalues()]

# delineate the workspace boundary
# for a in xrange(2):
#     for b in xrange(2):
#         x = a * workspace_low[0] + (1. - a) * workspace_high[0]
#         y = b * workspace_low[0] + (1. - b) * workspace_high[0]
#         execute_linear(left, x, y, workspace_high[2], Quat.from_v_theta([0,0,1], 0.), n_steps=1)
# rospy.sleep(1.)

# execute_linear(left, 0.6, 0.7, z_initial, Quat.from_v_theta([0,0,1], 0.), n_steps=1)

# fitting a gaussian
dbscan = cluster.DBSCAN(15.)
meanshift = cluster.MeanShift(cluster_all=False)
n_successes = 0
n_ik_errors = 0
for i in xrange(n_attempts):
    print '* attempt %i' % i
    # move to a random location in workspace
    a, b, c = np.random.rand(3)
    x = a * workspace_low[0] + (1. - a) * workspace_high[0]
    y = b * workspace_low[0] + (1. - b) * workspace_high[0]
    execute_linear(left, x, y, z_initial, Quat.from_v_theta([0,0,1], c * 2. * np.pi), n_steps=1)

    rospy.sleep(1.)
    # localize one cube
    # cu, cv = localize_cm_method(current_image, gripper_mask)
    cu, cv = localize_cluster_method(current_image, gripper_mask, dbscan, percentile=3)
    # cu, cv = localize_cluster_method(current_image, gripper_mask, meanshift)

    nx, ny, nz = camera.projectPixelTo3dRay((cu-320, cv-200))
    # get transform at the time of image acquisition
    obj_point = PointStamped()
    obj_point.header.frame_id = camera.tfFrame()
    obj_point.header.stamp = current_image_stamp
    obj_point.point.x = nx
    obj_point.point.y = ny
    obj_point.point.z = nz

    pt = tl.transformPoint('/base', obj_point)

    # find ray-table intersection
    camera_xyz, _ = tl.lookupTransform('/base', camera.tfFrame(), current_image_stamp)
    k = (z_table - camera_xyz[2]) / (pt.point.z - camera_xyz[2])
    # intersection: k * pt.point.x, k * pt.point.y, k * pt.point.z
    ox, oy = k * (pt.point.x - camera_xyz[0]) + camera_xyz[0], k * (pt.point.y - camera_xyz[1]) + camera_xyz[1]
    print 'object location', ox, oy

    # u, v = camera.project3dToPixel((ox, oy, z_table))
    # u, v = camera.project3dToPixel((nx, ny, nz))

    # display_image = PIL_Image.open('gripper_mask.png').convert('L')

    display_image = current_image.copy()
    draw = PIL_ImageDraw.Draw(display_image)
    draw.ellipse([(cu-10, cv-10), (cu+10, cv+10)], outline=(0, 255, 0))
    # draw.ellipse([(u+320-10, v+200-10), (u+320+10, v+200+10)], outline=(0, 255, 0))
    coords = map(lambda x: '%g' % x, left.endpoint_pose()['position'])
    quat = map(str, left.endpoint_pose()['orientation'])
    draw.text((20, 40), 'hand: '+', '.join(coords), font=font)
    draw.text((20, 20), 'object: '+', '.join(['%g' % ox, '%g' % oy]), font=font)
    msg = Image(
        header=Header(
            stamp=rospy.Time.now(),
            frame_id='base',
        ),
        width=640,
        height=400,
        step=640 * 4,
        encoding='bgra8',
        is_bigendian=0,
        data=display_image.tobytes(),
    )
    pub.publish(msg)

    # choose a random
    quat = Quat.from_v_theta([0., 0., 1.], np.random.rand() * 2. * np.pi)
    # move to pre-grasp pose
    execute_linear(left, ox, oy, z_initial, quat, n_steps=1)
    # IK execution reached pose from the commanded pose
    execution_error = np.linalg.norm(np.asarray(left.endpoint_pose()['position'])[:2] - [ox, oy])
    print 'execution error:', execution_error
    if execution_error < execution_error_threshold:
        # execute vertical pinch grasp
        traj = execute_vertical(left, z_table, n_steps=8, duration=4., timeout=8.)
        left_gripper.close()
        rospy.sleep(0.5)
        # lift arm
        execute_vertical(left, z_initial, n_steps=1, duration=2., timeout=4.)
        rospy.sleep(0.5)
        # use grasp force to recognize grasp success
        print 'grasp force', force
        if force > grasp_force_threshold:
            print 'grasp succeeded'
            # move to a random location in workspace
            a, b, c = np.random.rand(3)
            x = a * workspace_low[0] + (1. - a) * workspace_high[0]
            y = b * workspace_low[0] + (1. - b) * workspace_high[0]
            execute_linear(left, x, y, z_initial, Quat.from_v_theta([0,0,1], c * 2. * np.pi), n_steps=1, duration=2., timeout=4.)
            n_successes += 1
        else:
            print 'grasp failed'
        # release object
        left_gripper.open()
    else:
        print 'IK trajectory execution error'
        n_ik_errors += 1
        continue


print '* summary'
print '# of attempts', n_attempts
print '# of successes', n_successes
print '# of failures', n_attempts - n_successes
print '# of IK errors', n_ik_errors
