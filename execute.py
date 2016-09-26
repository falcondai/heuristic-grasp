import rospy
import baxter_interface
from baxter_interface.limb import Limb
from baxter_pykdl import baxter_kinematics
import numpy as np
import tqdm
from copy import deepcopy

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

import sys, os, time, cPickle, json, argparse

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
    draw.ellipse([(best_cu-10, best_cv-10), (best_cu+10, best_cv+10)], outline=(0, 0, 255))
    debug_image.save('clusters.png')
    return best_cu, best_cv

def tf_uv2xy(camera, tl, z_table, stamp, cu, cv):
    nx, ny, nz = camera.projectPixelTo3dRay((cu-320, cv-200))
    # get transform at the time of image acquisition
    obj_point = PointStamped()
    obj_point.header.frame_id = camera.tfFrame()
    obj_point.header.stamp = stamp
    obj_point.point.x = nx
    obj_point.point.y = ny
    obj_point.point.z = nz

    pt = tl.transformPoint('/base', obj_point)

    # find ray-table intersection
    camera_xyz, _ = tl.lookupTransform('/base', camera.tfFrame(), stamp)
    k = (z_table - camera_xyz[2]) / (pt.point.z - camera_xyz[2])
    # intersection: k * pt.point.x, k * pt.point.y, k * pt.point.z
    ox, oy = k * (pt.point.x - camera_xyz[0]) + camera_xyz[0], k * (pt.point.y - camera_xyz[1]) + camera_xyz[1]
    # print 'object location', ox, oy
    return ox, oy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_attempts', type=int, default=10)
    parser.add_argument('--no_replacement', action='store_true')
    parser.add_argument('--two_shot', action='store_true')

    args = parser.parse_args()

    rospy.init_node('heuristic_grasp')
    rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
    rs.enable()

    # task setup
    execution_error_threshold = 0.04
    # z_initial = -0.05
    # z_table = -0.165
    z_initial = 0.
    grasp_force_threshold = 20.
    workspace_margin = 0.08
    wx, wy, wz = json.load(open('marker.0.json'))['position']
    ax, ay, az = json.load(open('marker.1.json'))['position']
    print '* workspace', wx, wy, wz
    if args.no_replacement:
        print '* no replacement and auxiliary workspace', ax, ay, az
    workspace_low = np.asarray([wx-workspace_margin, wy-workspace_margin])
    workspace_high = np.asarray([wx+workspace_margin, wy+workspace_margin])
    aux_workspace_low = np.asarray([ax-workspace_margin, ay-workspace_margin])
    aux_workspace_high = np.asarray([ax+workspace_margin, ay+workspace_margin])
    z_table = wz

    # set up camera
    cc = baxter_interface.CameraController('left_hand_camera')
    cc.resolution = (640, 400)
    msg = rospy.wait_for_message('/cameras/left_hand_camera/camera_info', CameraInfo)
    camera = PinholeCameraModel()
    camera.fromCameraInfo(msg)

    # arms
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

    dbscan = cluster.DBSCAN(15.)
    meanshift = cluster.MeanShift(cluster_all=False)
    n_successes = 0
    n_ik_errors = 0
    for i in xrange(args.n_attempts):
        print '* attempt %i' % i
        # move to a random location in workspace
        a, b, c = np.random.rand(3)
        x = a * workspace_low[0] + (1. - a) * workspace_high[0]
        y = b * workspace_low[1] + (1. - b) * workspace_high[1]
        print 'moving to a random location', x, y
        execute_linear(left, x, y, z_initial, Quat.from_v_theta([0,0,1], c * 2. * np.pi), n_steps=1)

        rospy.sleep(1.)
        # localize one cube
        # cu, cv = localize_cm_method(current_image, gripper_mask)
        stamp = deepcopy(current_image_stamp)
        cu, cv = localize_cluster_method(current_image, gripper_mask, dbscan, percentile=3)
        # cu, cv = localize_cluster_method(current_image, gripper_mask, meanshift)

        # transform from image UV frame to base XY frame
        ox, oy = tf_uv2xy(camera, tl, z_table, stamp, cu, cv)
        print 'object location', ox, oy

        display_image = current_image.copy()
        draw = PIL_ImageDraw.Draw(display_image)
        draw.ellipse([(cu-10, cv-10), (cu+10, cv+10)], outline=(0, 255, 0))
        # draw.ellipse([(u+320-10, v+200-10), (u+320+10, v+200+10)], outline=(0, 255, 0))
        coords = map(lambda x: '%g' % x, left.endpoint_pose()['position'])
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

        # choose a random gripper orientation
        quat = Quat.from_v_theta([0., 0., 1.], np.random.rand() * 2. * np.pi)
        # move to pre-grasp pose
        execute_linear(left, ox, oy, z_initial, quat, n_steps=1)
        if args.two_shot:
            rospy.sleep(0.5)
            stamp = deepcopy(current_image_stamp)
            display_image = current_image.copy()
            cu, cv = localize_cluster_method(current_image, gripper_mask, dbscan, percentile=3)
            ox, oy = tf_uv2xy(camera, tl, z_table, stamp, cu, cv)
            print '2nd shot, object location', ox, oy

            draw = PIL_ImageDraw.Draw(display_image)
            draw.ellipse([(cu-10, cv-10), (cu+10, cv+10)], outline=(0, 255, 0))
            # draw.ellipse([(u+320-10, v+200-10), (u+320+10, v+200+10)], outline=(0, 255, 0))
            coords = map(lambda x: '%g' % x, left.endpoint_pose()['position'])
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

            execute_linear(left, ox, oy, z_initial, quat, n_steps=1)

        # IK execution reached pose from the commanded pose
        execution_error = np.linalg.norm(np.asarray(left.endpoint_pose()['position'])[:2] - [ox, oy])
        print 'execution error:', execution_error
        if execution_error < execution_error_threshold:
            # execute vertical pinch grasp
            traj = execute_vertical(left, z_table, n_steps=8, duration=4., timeout=8.)
            left_gripper.close()
            rospy.sleep(0.5)
            if left.endpoint_pose()['position'][2] - z_table > 0.5 * (z_initial - z_table):
                print 'IK descent trajectory execution error'
                n_ik_errors += 1
            else:
                # lift arm
                execute_vertical(left, z_initial, n_steps=1, duration=2., timeout=4.)
                rospy.sleep(0.5)
                # use grasp force to recognize grasp success
                print 'grasp force', force
                if force > grasp_force_threshold:
                    print 'grasp succeeded'
                    # move to a random location in workspace
                    a, b, c = np.random.rand(3)
                    if args.no_replacement:
                        x = a * aux_workspace_low[0] + (1. - a) * aux_workspace_high[0]
                        y = b * aux_workspace_low[1] + (1. - b) * aux_workspace_high[1]
                    else:
                        x = a * workspace_low[0] + (1. - a) * workspace_high[0]
                        y = b * workspace_low[1] + (1. - b) * workspace_high[1]
                    print 'dropping the grasped object at a random location', x, y
                    execute_linear(left, x, y, z_initial, Quat.from_v_theta([0,0,1], c * 2. * np.pi), n_steps=1, duration=2., timeout=4.)
                    n_successes += 1
                else:
                    print 'grasp failed'
            # release object
            left_gripper.open()
        else:
            print 'IK approach trajectory execution error'
            n_ik_errors += 1
            continue


    print '* summary'
    print '# of attempts', args.n_attempts
    print '# of successes', n_successes
    print '# of failures', args.n_attempts - n_successes
    print '# of IK errors', n_ik_errors
