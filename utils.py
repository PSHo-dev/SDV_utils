# BSD 3-Clause License
# 
# Copyright (c) [2025], [Ping Shu Ho]
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import carb
import omni.usd
import omni.kit.commands
import omni.timeline
from pxr import UsdGeom, Gf, Vt
import math
import numpy as np
import time
from matplotlib import cm

# os Related
def get_textures(dir_path):
    textures = []
    for file in os.listdir(dir_path):
        if file.endswith(".png"):
            textures.append(dir_path + file)
    return textures

# Matrix Computation
def normalize_vector(vector):
    if vector.ndim==1:
        norm = np.linalg.norm(vector, axis=0, keepdims=True)
    else:
        norm = np.linalg.norm(vector, axis=1, keepdims=True)

    norm[norm == 0] = 1 # Replace zero norms with 1 to avoid division by zero
    return vector / norm

def vector2magnitude(v):
    '''
    Computing the Magnitude of a provided 3D Vector
    Input: 3D Vector (N,3)
    Output: Magnitude of 3D Vector (N,1)
    '''
    components = v[:, 0:3]
    magnitude_square = np.einsum('ij,ij->i', components, components, optimize=True)
    magnitude = np.sqrt(magnitude_square)

    return magnitude

def quaternion_from_upvector(v):
    # vectors shape (N, 3)
    u = np.array([0, 0, 1], dtype=np.float64)
    # Cross product of each vector with u
    w = np.cross(np.tile(u, (v.shape[0], 1)), v)
    # Dot product of each vector with u
    dot = np.dot(v, u)
    # Compose quaternion array (w, x, y, z)
    q = np.empty((v.shape[0], 4), dtype=np.float64)
    q[:, 0] = dot
    q[:, 1:] = w
    # Add sqrt of sum of squares of q elements to q[:, 0]
    q[:, 0] += np.sqrt(np.sum(q**2, axis=1))
    # Normalize quaternions
    q_norm = normalize_vector(q)
    return q_norm

def create2Dmeshgrid(x_axis, y_axis):
    '''
    Computing the Coordinates of 2D grid
    Input: x,y axes range, and specified interval
    Output: Coordinates of 2D grid
    '''
    # Define the axis ranges
    x = np.linspace(x_axis[0], x_axis[1], int(x_axis[2]))
    y = np.linspace(y_axis[0], y_axis[1], int(y_axis[2]))
    # Create the 3D meshgrid
    X, Y= np.meshgrid(x, y, indexing='xy')
    # Convert to the grid coordinates
    coords = np.concatenate((X[:,:,None], Y[:,:,None]), axis=-1)
    positions = coords.reshape(int(x_axis[2]*y_axis[2]), -1)

    return positions

def create3Dmeshgrid(x_axis, y_axis, z_axis):
    '''
    Computing the Coordinates of 3D grid
    Input: x,y,z axes range, and specified interval
    Output: Coordinates of 3D grid
    '''
    # Define the axis ranges
    x = np.linspace(x_axis[0], x_axis[1], int(x_axis[2]))
    y = np.linspace(y_axis[0], y_axis[1], int(y_axis[2]))
    z = np.linspace(z_axis[0], z_axis[1], int(z_axis[2]))
    # Create the 3D meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
    # Convert to the grid coordinates
    coords = np.concatenate((X[:,:,:,None], Y[:,:,:, None], Z[:,:,:, None]), axis=-1)
    positions = coords.reshape(int(x_axis[2]*y_axis[2]*z_axis[2]), -1)

    return positions

# pxr, omni.usd Related
def get_current_stage():
    context = omni.usd.get_context()
    stage = context.get_stage()
    return stage

def start_timeline():
    timeline_interface = omni.timeline.get_timeline_interface()
    timeline_interface.play()

def check_path(path: str):
    if not path:
        carb.log_error("No path was given")
        return False
    return True

def get_prim(prim_path: str):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    return prim

def is_valid_prim(path: str):
    prim = get_prim(path)
    if not prim.IsValid():
        carb.log_warn(f"No valid prim at path given: {path}")
        return None
    return prim

def delete_prim(path: str):
    omni.kit.commands.execute('DeletePrims',
        paths=[path],
        destructive=False)

def get_prim_attr(prim_path: str, attr_name: str):
    prim = get_prim(prim_path)
    return prim.GetAttribute(attr_name).Get()

def set_prim_attr(prim_path: str, attr_name: str, new_value):
    prim = get_prim(prim_path)
    prim.GetAttribute(attr_name).Set(new_value)

def get_up_axis():
    stage = get_current_stage()
    up_axis = UsdGeom.GetStageUpAxis(stage)
    return up_axis

def realign_prim_to_target(target_path, prim_path):
    position = get_prim_attr(target_path, 'xformOp:translate')
    rotation = get_prim_attr(target_path, 'xformOp:rotateXYZ') or Gf.Vec3d(0,0,0)
    set_prim_attr(prim_path, 'xformOp:translate', position)
    set_prim_attr(prim_path, 'xformOp:rotateXYZ', rotation)
    proxy_world_trans = omni.usd.get_world_transform_matrix(get_current_stage().GetPrimAtPath(prim_path))
    proxy_rotation = proxy_world_trans.ExtractRotationQuat().GetNormalized()
    y_axis = (0,1,0)
    up_vector = (proxy_rotation * Gf.Quatd(0, y_axis) * proxy_rotation.GetInverse()).GetImaginary()
    trans = (up_vector * 250) + get_prim_attr(prim_path, 'xformOp:translate')
    set_prim_attr(prim_path, 'xformOp:translate', trans)

def look_at(target_path, prim_path):
    forward = Gf.Vec3d(1.0,0.0,0.0)
    target = get_prim_attr(target_path, 'xformOp:translate')
    start = get_prim_attr(prim_path, 'xformOp:translate')
    direction = Gf.Vec3d(target) - start
    rotation = Gf.Rotation(forward, direction)
    decomposed = rotation.Decompose(Gf.Vec3d.ZAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.XAxis())
    rotateXYZ = (decomposed[2], decomposed[1], decomposed[0])
    set_prim_attr(prim_path, 'xformOp:rotateXYZ', rotateXYZ)

def colorArray_numpy_to_color3f(colorArray_numpy):
    # Convert the numpy array to a Color3fArray
    color3f_array = Vt.Vec3fArray(len(colorArray_numpy))
    for i, row in enumerate(colorArray_numpy):
        color3f_array[i] = tuple(row)

    return color3f_array

def create_vt_int_array(size):
    """Create a VtIntArray of a specified size.

    Args:
        size (int): The number of integers in the array.

    Returns:
        Vt.IntArray: A VtIntArray with the specified size.
    """
    vt_array = Vt.IntArray(size)
    for i in range(size):
        vt_array[i] = i
    return vt_array

def update_visiblity(fieldPrim, visible):
    fieldPrim_visibility = fieldPrim.GetAttribute("visibility")
    if visible:
        fieldPrim_visibility.Set("inherited")
    else:
        fieldPrim_visibility.Set("invisible")

def compute_faceVertex_color(vertex_intensity, face_vertex_indices, colormap_name):
    # Load Matplotlib colormap
    colormap = cm.get_cmap(colormap_name)

    # Convert vertex intensities to face-varying intensities by indexing
    face_varying_intensity = vertex_intensity[face_vertex_indices]

    # Map normalized intensity to RGB colors using viridis colormap
    colors_np = colormap(face_varying_intensity)[:, :3]  # discard alpha channel

    # Convert numpy colors to list of Gf.Vec3f for USD
    colors = [Gf.Vec3f(*rgb) for rgb in colors_np]

    return colors

def compute_plane_vertex(grid_x, grid_Y):
    # Define quads topology
    face_vertex_counts = []
    face_vertex_indices = []
    for y in range(grid_Y - 1):
        for x in range(grid_x - 1):
            face_vertex_counts.append(4)
            v0 = y * grid_x + x
            v1 = v0 + 1
            v2 = v1 + grid_x
            v3 = v0 + grid_x
            face_vertex_indices.extend([v0, v1, v2, v3])

    face_vertex_indices = [int(x) for x in face_vertex_indices]

    return face_vertex_counts, face_vertex_indices

def modify_visibility_of_instances(point_instancer_path, invisible_ids_np=None, all_visible=False):
    stage = omni.usd.get_context().get_stage()
    point_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(point_instancer_path))
    invisible_ids_attr = point_instancer.GetInvisibleIdsAttr()
    invisible_ids = invisible_ids_attr.Get()

    if all_visible == True:
        invisible_ids = Vt.Int64Array([])
        invisible_ids_attr.Set(Vt.Int64Array(invisible_ids))
    elif type(invisible_ids_np)==type(None) and  all_visible==False:
        pass
    else:
        invisible_ids = Vt.Int64Array(invisible_ids_np.tolist())
        invisible_ids_attr.Set(Vt.Int64Array(invisible_ids))

def compute_invisible_ids(grid_xy, layers, visible_layer_id):
    '''
    grid_xy [x,y]
    layers [z]
    '''

    grid_size = grid_xy[0]*grid_xy[1]*layers
    # Create zero array of grid size
    all_invisible = np.zeros(grid_size, dtype=np.int64)

    #visible_indices = np.arange(0, grid_size, layers,  dtype=np.int64) + int(visible_layer_id)
    visible_indices = np.arange(grid_xy[0]*grid_xy[1]*(visible_layer_id), grid_xy[0]*grid_xy[1]*(visible_layer_id+1), dtype=np.int64)

    # Modify values at visible indices to 1
    bool_visible = all_invisible
    bool_visible[visible_indices] = int(1)

    # Get all zero indices as invisible ids
    invisible_ids_np = np.nonzero(bool_visible==0)[0]

    return invisible_ids_np

def modify_size_of_glyph(glyph_path, scale_value):
    stage = omni.usd.get_context().get_stage()

    glyph_prim = stage.GetPrimAtPath(glyph_path)

    if not glyph_prim.IsValid():
        raise ValueError(f"Prim not found at path: {prim_path}")
    
    scale_att = glyph_prim.GetAttribute("xformOp:scale")
    scale_att.Set(Gf.Vec3f(scale_value, scale_value, scale_value))


def modify_visibility_of_children(primPath, visible_index=None, all_visible=False):
    stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath(primPath)
    primChildrenList = list(prim.GetChildren())

    if all_visible == True:
        for primChild in primChildrenList:
           update_visiblity(primChild, True)
    elif type(visible_index)==type(None) and  all_visible==False:
        pass
    else:
        for primChild in primChildrenList:
            update_visiblity(primChild, False)
        update_visiblity(primChildrenList[visible_index], True)