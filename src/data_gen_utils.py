
import mesh_to_sdf

import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

import sys
sys.path.append('../../../rai-fork/rai/ry')
import libry as ry


def angle(v1, v2):
    a = np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)).clip(-1, 1)
    angle = np.arccos(a)*180/np.pi
    return angle

def get_rotation_matrix(angle, axis):
    matrix = np.identity(4)
    matrix[:3, :3] = Rotation.from_euler(axis, angle).as_matrix()
    return matrix
        
def get_equidistant_camera_angles(total_num_cam):
    indices = np.arange(0, total_num_cam, dtype=float) + 0.5
    theta = np.arccos(1 - 2*indices/total_num_cam)
    phi = (np.pi * (1 + 5**0.5) * indices) % (2 * np.pi)
    return phi, theta

def get_camera_transform(cam_pos, origin):
    rel_pos = cam_pos - origin
    theta = np.arctan2(np.linalg.norm(rel_pos[:2]), rel_pos[2])
    phi = np.pi/2 + np.arctan2(rel_pos[1], rel_pos[0])
    distance = np.linalg.norm(rel_pos)
    T = get_camera_transform_looking_at_origin(phi, theta, distance)
    T[:3,3] += origin
    
    return T

def get_camera_transform_looking_at_origin(phi, theta, camera_distance):
    camera_transform = np.identity(4)
    camera_transform[2, 3] = camera_distance
    camera_transform = np.matmul(get_rotation_matrix(theta, axis='x'), camera_transform)
    camera_transform = np.matmul(get_rotation_matrix(phi, axis='z'), camera_transform)
    return camera_transform

def get_camera_projection_matrix(cam_distance, r=1.):
    fov = 2*np.arcsin(r/cam_distance)
    cam_projection_matrix = np.zeros((4,4))
    cam_projection_matrix[0,0] = 1/np.tan(fov/2)
    cam_projection_matrix[1,1] = -1/np.tan(fov/2)
    cam_projection_matrix[2,2] = -1/r
    cam_projection_matrix[2,3] = -cam_distance/r
#     cam_projection_matrix[2,2] = -cam_distance/r
#     cam_projection_matrix[2,3] = -(cam_distance**2-r**2)/r
    cam_projection_matrix[3,2] = -1.
    return cam_projection_matrix

# allows for varying center
# TODO: should revert it back!!
def render_mesh_rai2(mesh_filename, 
                     num_cam = 10, 
                     mu_cam_distance = 1.5, 
                     sig_cam_distance = .3, 
                     obj_radius = .1, 
                     mu_obj_center = np.zeros(3),
                     sig_obj_center = 0.01,
                     res = 100,
                     view = False):

    C = ry.Config()
    camera = C.cameraView()
    mug = C.addMeshFrame(mesh_filename, 'mug')
    quat0 = mug.getQuaternion()

    if view: plt.figure(figsize=(20,20))
    phi, theta = get_equidistant_camera_angles(num_cam)
    cam_trans_inv_list = []
    cam_projection_list = []
    rgb_list = []
    for i, (phi, theta) in enumerate(zip(phi,theta)):
        # print(i, np.rad2deg(phi), np.rad2deg(theta))
        cam_distance = mu_cam_distance + np.random.randn(1)*sig_cam_distance
        camera_transform = get_camera_transform_looking_at_origin(phi, theta, cam_distance)
        
        obj_center = mu_obj_center + np.random.randn(1)*sig_obj_center
        camera_transform[:3,3] += obj_center
        P = get_camera_projection_matrix(cam_distance, obj_radius)

        cam_name = 'camera'+str(i)
        C.addFrame(cam_name, 'mug').setShape(ry.ST.marker, [0.001])\
                                   .setRelativeAffineMatrix(camera_transform)
        
        random_pos = np.array([0,0,.9]) + np.random.randn(3)*np.array([.5,.5,.1])
        random_quat = np.random.randn(4); random_quat /= np.linalg.norm(random_quat)
        mug.setQuaternion(random_quat).setPosition(random_pos) # randomizes lighting
        
        camera.updateConfig(C)
        camera.addSensor(cam_name, cam_name, res, res, 0.5*P[0,0])
        rgb, depth = camera.computeImageAndDepth()
        mask = camera.extractMask('mug')
        mug.setQuaternion(quat0)
        rgb *= np.expand_dims(mask,axis=2)

        rgb_list.append(rgb)
        cam_trans_inv_list.append(np.linalg.inv(camera_transform))
        cam_projection_list.append(P)

        if view: 
            ax = plt.subplot(int(np.ceil(num_cam/10)),10,i+1)
            ax.imshow(rgb)

    if view: 
        plt.show()
        V = ry.ConfigurationViewer()
        V.setConfiguration(C)
        input()

    return np.stack(rgb_list), np.stack(cam_trans_inv_list), np.stack(cam_projection_list)


def render_mesh_rai(mesh_filename, 
                    num_cam = 10, 
                    cam_distance_center = 15, 
                    mu_cam_distance = 1.5, 
                    sig_cam_distance = .3, 
                    obj_radius = 1., 
                    obj_center = np.zeros(3),
                    res = 100,
                    view = False):

    C = ry.Config()
    camera = C.cameraView()
    mug = C.addMeshFrame(mesh_filename, 'mug')
    quat0 = mug.getQuaternion()

    if view: plt.figure(figsize=(20,20))
    phi, theta = get_equidistant_camera_angles(num_cam)
    cam_trans_inv_list = []
    cam_projection_list = []
    rgb_list = []
    for i, (phi, theta) in enumerate(zip(phi,theta)):
        cam_distance = mu_cam_distance + np.random.randn(1)*sig_cam_distance
        camera_transform = get_camera_transform_looking_at_origin(phi, theta, cam_distance)
        camera_transform[:3,3] += obj_center
        P = get_camera_projection_matrix(cam_distance, obj_radius)

        cam_name = 'camera'+str(i)
        C.addFrame(cam_name, 'mug').setShape(ry.ST.marker, [0.0])\
                                   .setRelativeAffineMatrix(camera_transform)
        random_pos = np.array([0,0,1.9]) + np.random.randn(3)*np.array([.5,.5,.1])
        random_quat = np.random.randn(4); random_quat /= np.linalg.norm(random_quat)
        mug.setQuaternion(random_quat).setPosition(random_pos) # randomizes lighting
        
        camera.updateConfig(C)
        camera.addSensor(cam_name, cam_name, res, res, 0.5*P[0,0])
        rgb, depth = camera.computeImageAndDepth()
        mask = camera.extractMask('mug')
        mug.setQuaternion(quat0)
        rgb *= np.expand_dims(mask,axis=2)

        rgb_list.append(rgb)
        cam_trans_inv_list.append(np.linalg.inv(camera_transform))
        cam_projection_list.append(P)

        if view: 
            ax = plt.subplot(int(np.ceil(num_cam/10)),10,i+1)
            ax.imshow(rgb)

    if view: 
        plt.show()
#         V = ry.ConfigurationViewer()
#         V.setConfiguration(C)
#         input()

    return np.stack(rgb_list), np.stack(cam_trans_inv_list), np.stack(cam_projection_list)



def render_mesh(mesh_filename, 
                num_cam = 10, 
                cam_distance_center = 15, 
                sig_cam_distance = 3., 
                obj_radius = 1., 
                obj_center = np.zeros(3),
                res = 100,
                view = False):

    obj_trimesh = trimesh.load(mesh_filename)
    obj_mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
    cam = pyrender.PerspectiveCamera(yfov=2*np.arcsin(obj_radius/cam_distance_center), aspectRatio=1.0)
    point_l = pyrender.PointLight(color=np.ones(3), intensity=3*cam_distance_center**2)

    scene = pyrender.Scene(bg_color=np.zeros(3), ambient_light=0.1*np.ones(3))
    obj_node = scene.add(obj_mesh)

    point_l_node = scene.add(point_l)
    cam_node = scene.add(cam)

    r = pyrender.OffscreenRenderer(viewport_width=res, viewport_height=res)

    if view: plt.figure(figsize=(20,20))
    phi, theta = get_equidistant_camera_angles(num_cam)
    cam_trans_inv_list = []
    cam_projection_list = []
    rgb_list = []
    for i, (phi, theta) in enumerate(zip(phi,theta)):
        # print(i, np.rad2deg(phi), np.rad2deg(theta))
        cam_distance = (cam_distance_center + np.random.randn(1)*sig_cam_distance)*obj_radius
        camera_transform = get_camera_transform_looking_at_origin(phi, theta, cam_distance)
        camera_transform[:3,3] += obj_center

        cam_node.camera.yfov = 2*np.arcsin(obj_radius/cam_distance)
        cam_node.matrix = camera_transform

        point_l_node.light.intensity = 3*cam_distance**2
        point_l_node.matrix = camera_transform
        rgb, depth = r.render(scene)

        rgb_list.append(rgb)
        cam_trans_inv_list.append(np.linalg.inv(camera_transform))
        cam_projection_list.append(get_camera_projection_matrix(cam_distance, obj_radius))

        if view: 
            ax = plt.subplot(int(np.ceil(num_cam/10)),10,i+1)
            ax.imshow(rgb)

    r.delete()
    if view: plt.show()

    return np.stack(rgb_list), np.stack(cam_trans_inv_list), np.stack(cam_projection_list)


def compute_sdf(obj_trimesh, N=5000, sig=0.01, scale=1., center=np.zeros(3), view=False):

    surface_points = obj_trimesh.sample(N)
#     unit_samples = sample_uniform_points_in_unit_sphere(N//5)*scale + center
    global_samples = np.random.randn(N//2,3)*scale + center.reshape(1,3)
    
    points = np.concatenate([
                surface_points + np.random.randn(N,3)*sig*scale, 
                surface_points + np.random.randn(N,3)*sig*scale*10,
                global_samples], axis = 0)

    sdf = trimesh.proximity.signed_distance(obj_trimesh, points)

    if view:
        colors = np.zeros(points.shape)
        colors[sdf < 0, 2] = 1
        colors[sdf > 0, 0] = 1
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_points(points, colors=colors))
        pyrender.viewer.Viewer(scene, use_raymond_lighting=True, point_size=10.)
        
    return points, sdf

def compute_sdf2(obj_trimesh, N=5000, sig=0.01, scale=1., center=np.zeros(3), view=False):

    obj_trimesh.apply_scale(1/scale)
    point_cloud = mesh_to_sdf.get_surface_point_cloud(obj_trimesh)
    
    
    surface_points = point_cloud.get_random_surface_points(N)
    global_samples = np.random.randn(N//2,3) + center.reshape(1,3)
    
    points = np.concatenate([
                surface_points + np.random.randn(N,3)*sig, 
                surface_points + np.random.randn(N,3)*sig*10,
                global_samples
    ], axis = 0)

    sdf = point_cloud.get_sdf_in_batches(points, 
                                         use_depth_buffer=True, 
                                         sample_count=10000000)
    
    if view:
        colors = np.zeros(points.shape)
        colors[sdf < 0, 2] = 1
        colors[sdf > 0, 0] = 1
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_points(points, colors=colors))
        pyrender.viewer.Viewer(scene, use_raymond_lighting=True, point_size=10.)
        
    return points*scale, sdf*scale