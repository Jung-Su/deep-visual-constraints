import torch
import numpy as np
from typing import Optional
import trimesh
from scipy.spatial.transform import Rotation




def view_scene_hang(pose7d, filename):
    pose7d = pose7d.detach().cpu().view(-1, 7).numpy()
    
    mug_meshes = []
    hook_meshes = []
    
    sn = int(np.sqrt(pose7d.shape[0]))
    for n in range(pose7d.shape[0]):
        
        T = np.eye(4)
        T[:3,:3] = Rotation.from_quat(pose7d[n,[4,5,6,3]]).as_matrix()
        T[:3,3] = pose7d[n,:3]
        Tinv = np.linalg.inv(T)
        mug_mesh = trimesh.load(filename).apply_transform(Tinv)
        
        T2 = np.eye(4)
        T2[0,3] = .4*(n//sn - 0.5*(sn-1))
        T2[1,3] = .4*(n%sn - 0.5*(sn-1))
        mug_meshes.append(mug_mesh.apply_transform(T2))
        hook_meshes.append(create_hook().apply_transform(T2))
        
    axis_mesh = trimesh.creation.axis(origin_size=0.004,
                                      axis_radius=0.003,
                                      axis_length=0.03)
    return trimesh.Scene(mug_meshes + hook_meshes + [axis_mesh])

def create_hook(color=[100, 200, 100], cylinder_radius=0.002):
    pillar = trimesh.creation.cylinder(
        radius=cylinder_radius*2,
        segment=[
            [0, 0.1, -0.15],
            [0, -0.2, -0.15],
        ],
    )
    hook = trimesh.creation.cylinder(radius=cylinder_radius, 
                                     height=.15*2)
    tmp = trimesh.util.concatenate([pillar, hook])
    tmp.visual.vertex_colors = color
    return tmp


def to_device(data, device):
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    return data

def view_scene_hang_batch(pose7d_batch, feasibility_batch, filename_list):
    """
    Args:
    pose7d_batch: (B, N, 7)
    feasibility_batch: (B, N)
    filename_list: (B,)
    """
    
    B, N, _ = pose7d_batch.shape
    mug_meshes = []
    hook_meshes = []
    
    sn = int(np.sqrt(len(filename_list)))
    for b, filename in enumerate(filename_list):
        
        mesh_filename = 'data/meshes_coll/'+filename
        T2 = np.eye(4)
        T2[0,3] = .4*(b//sn - 0.5*(sn-1))
        T2[1,3] = .4*(b%sn - 0.5*(sn-1))
        hook_meshes.append(create_hook().apply_transform(T2))
        
        for n in range(N):
            pose7d = pose7d_batch[b,n]
            T = np.eye(4)
            T[:3,:3] = Rotation.from_quat(pose7d[[4,5,6,3]]).as_matrix()
            T[:3,3] = pose7d[:3]
            Tinv = np.linalg.inv(T)
            mug_mesh = trimesh.load(mesh_filename).apply_transform(Tinv)
            if not feasibility_batch[b,n]:
                mug_mesh.visual.vertex_colors = [200, 100, 100]

            mug_meshes.append(mug_mesh.apply_transform(T2))
        
        
    axis_mesh = trimesh.creation.axis(origin_size=0.004, 
                                      axis_radius=0.003, 
                                      axis_length=0.03)
    return trimesh.Scene(mug_meshes + hook_meshes + [axis_mesh])

def view_scene_grasp_batch(pose7d_batch, feasibility_batch, filename_list, draw_coll=False):
    """
    Args:
    pose7d_batch: (B, N, 7)
    feasibility_batch: (B, N)
    filename_list: (B,)
    """
    
    B, N, _ = pose7d_batch.shape
    
    mug_meshes = []
    gripper_meshes = []
    
    sn = int(np.sqrt(len(filename_list)))
    for b, filename in enumerate(filename_list):
        mesh_filename = 'data/meshes_coll/'+filename
        T2 = np.eye(4)
        T2[0,3] = .4*(b//sn - 0.5*(sn-1))
        T2[1,3] = .4*(b%sn - 0.5*(sn-1))
        mug_meshes.append(trimesh.load(mesh_filename).apply_transform(T2))
        
        for n in range(N):
            pose7d = pose7d_batch[b,n]
            T = np.eye(4)
            T[:3,:3] = Rotation.from_quat(pose7d[[4,5,6,3]]).as_matrix()
            T[:3,3] = pose7d[:3]
            
            if draw_coll:
                gripper_mesh = create_gripper_coll().apply_transform(T)
            else:
                gripper_mesh = create_gripper_marker().apply_transform(T)

            if not feasibility_batch[b,n]:
                gripper_mesh.visual.vertex_colors = [200, 100, 100]

            gripper_meshes.append(gripper_mesh.apply_transform(T2))
        
    return trimesh.Scene(mug_meshes + gripper_meshes)

def view_scene_grasp(pose7d, filename):
    
    obj_mesh = trimesh.load(filename)
    pose7d = pose7d.detach().cpu().view(-1, 7).numpy()
    grippers_mesh = []
    for n in range(pose7d.shape[0]):
        T = np.eye(4)
        T[:3,:3] = Rotation.from_quat(pose7d[n,[4,5,6,3]]).as_matrix()
        T[:3,3] = pose7d[n,:3]
        grippers_mesh.append(create_gripper_marker().apply_transform(T))

    return trimesh.Scene([obj_mesh] + grippers_mesh)

def create_gripper_coll(color=[0, 0, 255, 120]):
    pose7d = np.array([0, 0, .064, -.5, .5, -.5, .5])
    T = np.eye(4)
    T[:3,:3] = Rotation.from_quat(pose7d[[4,5,6,3]]).as_matrix()
    T[:3,3] = pose7d[:3]
    T1 = np.eye(4)
    T1[2,3] = -.075

    coll0 = trimesh.creation.capsule(radius=0.03, height=.15).apply_transform(T@T1)

    coll1 = trimesh.creation.icosphere(radius=0.02)
    coll1.vertices += [-0.058, 0,  0]
    
    coll2 = trimesh.creation.icosphere(radius=0.02)
    coll2.vertices += [0.058, 0,  0]
    
    coll3 = trimesh.creation.icosphere(radius=0.02)
    coll3.vertices += [-0.07,  0,  0.0256]
    
    coll4 = trimesh.creation.icosphere(radius=0.02)
    coll4.vertices += [0.07,  0,  0.0256]
        
    tmp = trimesh.util.concatenate([coll0, coll1, coll2, coll3, coll4])
    tmp.visual.vertex_colors = color

    return tmp



def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.
    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.
    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002, segment=[[0.05, 0, -0.02], [0.05, 0, 0.045]],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002, segment=[[-0.05, 0, -0.02], [-0.05, 0, 0.045]],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, segment=[[0, 0, 0.045], [0, 0, 0.090]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002, segment=[[-0.05, 0, 0.045], [0.05, 0, 0.045]],
    )
    
    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.vertex_colors = color

    return tmp

def apply_delta(x, delta_x):
    """
    apply delta_x
    Input
        x: (B, 7, 1) pose
        delta_x: (B, 6, 1) delta

    Return (B, 7, 1)
    """
    y = torch.empty_like(x)
    y[:, :3] = x[:, :3] + delta_x[:, :3]

    delta_q = axis_angle_to_quaternion(delta_x[:, 3:, 0])
    y[:, 3:] = quaternion_multiply(x[:, 3:, 0], delta_q).view(-1, 4, 1)
    return y
    
############################################################################

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)

def axis_angle_to_matrix(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles
    
def standardize_quaternion(quaternions):
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)
    
def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_multiply(a, b):
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def quaternion_invert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * torch.tensor([1, -1, -1, -1], 
                                     dtype=quaternion.dtype, 
                                     device=quaternion.device)



def quaternion_apply(quaternion, point):
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)
    
def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.
        requires_grad: Whether the resulting tensor should have the gradient
            flag set.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = torch.randn((n, 4), dtype=dtype, device=device, requires_grad=requires_grad)
#     o = o.sign()*torch.where(o.abs() > 1e-4, o.abs(), 1e-4*torch.ones_like(o))
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s+1e-10), o[:, 0])[:, None]
    return o

def quaternion_slerp(q0, q1, t):
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        q0, q1: Tensor of quaternions, real part first, of shape (..., 4).
        t: interpolation parameter between 0 and 1 (..., 1).

    Returns:
        Tensor of rotated points of shape (..., 4).
    """
    
    q_tmp = quaternion_multiply(q1, quaternion_invert(q0))
    rot = t*quaternion_to_axis_angle(q_tmp)
    q_interp = quaternion_multiply(axis_angle_to_quaternion(rot), q0)
    
    return q_interp


def index(feat, uv):
    """
    Extract image features at uv coordinates
    
    Args:
        feat: (B, Feat_img, H, W) image features
        uv: (B, N, 2) uv coordinates in the image plane, range [-1, 1]
    
    Returns:
        (B, N, Feat_img) image features
    """
    samples = torch.nn.functional.grid_sample(feat, 
                                              uv.unsqueeze(2),  # (B, N, 1, 2) 
                                              align_corners=True, 
                                              padding_mode="border")# (B, Feat_img, N, 1)
        
    return samples.transpose(1,2).squeeze(-1)  # (B, N, Feat_img)


def perspective(points, projection_matrices):
    """
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    
    Args:
        points: (B, N, 3) Tensor of 3D points
        projection_matrices: (B, 4, 4) Tensor of projection matrix
        
    Returns:
        uv: (B, N, 2) uv coordinates in image space
        z: (B, N, 1) normalized depth
    """
    
    tmp = torch.ones_like(points[...,0:1])
    points = torch.cat([points, tmp], dim=2)  # (B, N, 4)
    
    homo = points.bmm(projection_matrices.transpose(1,2)) # (B, N, 4)
    
    uv, z, w = torch.split(homo, [2,1,1], dim=2)
    w = w.clamp(min=1e-2) # clamp depth near the camera (1 cm)

#     return uv/w, z/w # (B, N, 2), (B, N, 1)
    return uv/w, z # (B, N, 2), (B, N, 1)