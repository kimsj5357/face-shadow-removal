import os
import numpy as np
from PIL import Image
import scipy.io as sio
from skimage.transform import estimate_transform, warp
import torch
import torch.nn.functional as F
from torchvision import transforms

def render_shadow(img_path, mesh_path, img_size=256):
    # img = np.array(Image.open(img_path).convert('RGB'))
    mesh = sio.loadmat(mesh_path)

    # h, w, _ = img.shape

    verts = mesh['verts']
    trans_verts = mesh['trans_verts']
    # normal_images = mesh['normal_images']
    rendering = mesh['rendering']

    # azimuth, elevation = img_path[-12:-4].split('E')
    # azimuth = int(azimuth)
    # elevation = int(elevation)
    # a = azimuth * np.pi / 180.
    # e = elevation * np.pi / 180.
    # x = np.cos(e) * np.sin(-a)
    # y = np.sin(e)
    # z = np.cos(e) * np.cos(-a)
    #
    # light_intensities = np.array([[1, 1, 1]])
    # light_positions = np.array([[x, y, z]])
    #
    # # tform = get_tform(img, img_size=img_size)
    # # tform = torch.inverse(tform).transpose(1, 2)
    #
    # lights = np.concatenate([light_positions, light_intensities], -1)
    #
    # alpha_images = rendering[-1, :, :][None, :, :]
    # albedo_images = rendering[:3, :, :]
    # normal_images = rendering[9:12, :, :]
    # normal_images = np.transpose(normal_images, (1, 2, 0))
    #
    # shading = add_directionlight(normal_images.reshape([-1, 3]), lights)[0].numpy()
    # shading_images = shading.reshape([h, w, 3])
    # shading_images = np.transpose(shading_images, (2, 0, 1))
    # shaded_images = albedo_images * shading_images
    #
    # shaded_images = np.transpose(shaded_images, (1, 2, 0))

    return rendering

def get_shadow(coord, rendering, imsize=(256, 256)):
    b, _ = coord.shape

    # light_intensities = np.array([[1, 1, 1]])
    light_intensities = torch.tensor([[[1, 1, 1]]]).to(coord.device)
    light_intensities = light_intensities.repeat((b, 1, 1))
    light_positions = coord.reshape(b, 1, 3)

    # lights = np.concatenate([light_positions, light_intensities], -1)
    lights = torch.cat([light_positions, light_intensities], -1)

    alpha_images = rendering[:, -1, :, :][:, None, :, :]
    albedo_images = rendering[:, :3, :, :]
    normal_images = rendering[:, 9:12, :, :]
    # normal_images = np.transpose(normal_images, (1, 2, 0))
    normal_images = normal_images.permute((0, 2, 3, 1))
    normal_images = normal_images.reshape([b, -1, 3])

    shading = add_directionlight(normal_images, lights)
    shading_images = shading.reshape([-1, albedo_images.shape[2], albedo_images.shape[3], 3])
    # shading_images = np.transpose(shading_images, (2, 0, 1))
    shading_images = shading_images.permute((0, 3, 1, 2))
    shaded_images = albedo_images * shading_images

    # shaded_images = np.transpose(shaded_images, (1, 2, 0))
    # shaded_images = shaded_images.permute((0, 2, 3, 1))

    resize = transforms.Resize(imsize)
    shaded_images_resize = resize(shaded_images)

    return shaded_images_resize

def add_directionlight(normals, lights):
    '''
        normals: [bz, nv, 3]
        lights: [bz, nlight, 6]
    returns:
        shading: [bz, nv, 3]
    '''
    if isinstance(normals, np.ndarray):
        normals = torch.tensor(normals).unsqueeze(0)
    if isinstance(lights, np.ndarray):
        lights = torch.tensor(lights).unsqueeze(0)


    light_direction = lights[:, :, :3]
    light_intensities = lights[:, :, 3:]
    directions_to_lights = F.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3)
    # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
    # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
    normals_dot_lights = torch.clamp((normals[:, None, :, :] * directions_to_lights).sum(dim=3), 0., 1.)
    shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
    return shading.mean(1)

def get_tform(img, img_size=256):
    h, w, _ = img.shape
    src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
    DST_PTS = np.array([[0, 0], [0, img_size - 1], [img_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    dst_image = warp(img, tform.inverse, output_shape=(img_size, img_size))

    return tform



def decompose_code(code, num_dict):
    ''' Convert a flattened parameter vector to a dictionary of parameters
    code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
    '''
    code_dict = {}
    start = 0
    for key in num_dict:
        end = start+int(num_dict[key])
        code_dict[key] = code[:, start:end]
        start = end
        if key == 'light':
            code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
    return code_dict

# ---------------------------- process/generate vertices, normals, faces
def generate_triangles(h, w, margin_x=2, margin_y=5, mask = None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    #.
    # w*h
    triangles = []
    for x in range(margin_x, w-1-margin_x):
        for y in range(margin_y, h-1-margin_y):
            triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
            triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:,[0,2,1]]
    return triangles


# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                   vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                   vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                   vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals


def load_obj(obj_filename):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = torch.tensor(verts, dtype=torch.float32)
    uvcoords = torch.tensor(uvcoords, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.long); faces = faces.reshape(-1, 3) - 1
    uv_faces = torch.tensor(uv_faces, dtype=torch.long); uv_faces = uv_faces.reshape(-1, 3) - 1
    return (
        verts,
        uvcoords,
        faces,
        uv_faces
    )