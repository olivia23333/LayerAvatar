import numpy as np
import json
import os
import pickle
import torch
import trimesh
import smplx
from pytorch3d.ops import knn_points


def subdivide(vertices, faces, attributes=None, face_index=None):
    """
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those faces will
    be subdivided and their neighbors won't be modified making
    the mesh no longer "watertight."

    Parameters
    ----------
    vertices : (n, 3) float
      Vertices in space
    faces : (n, 3) int
      Indexes of vertices which make up triangular faces
    attributes: (n, d) float
      vertices attributes
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces

    Returns
    ----------
    new_vertices : (n, 3) float
      Vertices in space
    new_faces : (n, 3) int
      Remeshed faces
    """
    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c,3) int set of vertex indices
    faces = faces[face_index]

    # the (c, 3, 3) float set of points in the triangles
    triangles = vertices[faces]
    
    # the 3 midpoints of each triangle edge
    # stacked to a (3 * c, 3) float
    mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1], [1, 2], [2, 0]]])

    # for adjacent faces we are going to be generating
    # the same midpoint twice so merge them here
    mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
    unique, inverse = trimesh.grouping.unique_rows(mid)
    mid = mid[unique]
    mid_idx = inverse[mid_idx] + len(vertices)

    # the new faces with correct winding
    f = np.column_stack([faces[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))
    # add the 3 new faces per old face
    new_faces = np.vstack((faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    new_vertices = np.vstack((vertices, mid))

    if attributes is not None:
        tri_att = attributes[faces]
        mid_att = np.vstack([tri_att[:, g, :].mean(axis=1) for g in [[0, 1], [1, 2], [2, 0]]])
        mid_att = mid_att[unique]
        new_attributes = np.vstack((attributes, mid_att))
        return new_vertices, new_faces, new_attributes, unique

    return new_vertices, new_faces, unique

def write_obj(obj_name,
              vertices,
              faces,
              uvcoords=None,
              uvfaces=None,
              ):
    ''' Save 3D face model with texture. 
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    '''
    if os.path.splitext(obj_name)[-1] != '.obj':
        obj_name = obj_name + '.obj'
    # mtl_name = obj_name.replace('.obj', '.mtl')
    # texture_name = obj_name.replace('.obj', '.png')
    # material_name = 'FaceTexture'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    # if inverse_face_order:
    #     faces = faces[:, [2, 1, 0]]
    #     if uvfaces is not None:
    #         uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        # f.write('# %s\n' % os.path.basename(obj_name))
        # f.write('#\n')
        # f.write('\n')

        # if texture is not None:
        #     f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

        # write vertices
        # if colors is None:
        for i in range(vertices.shape[0]):
            f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        # else:
            # for i in range(vertices.shape[0]):
                # f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

        # write uv coords
        # if texture is None:
        #     for i in range(faces.shape[0]):
        #         f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
        # else:
        for i in range(uvcoords.shape[0]):
            f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
        # f.write('usemtl %s\n' % material_name)
        # write f: ver ind/ uv ind
        uvfaces = uvfaces + 1
        for i in range(faces.shape[0]):
            f.write('f {}/{} {}/{} {}/{}\n'.format(
                #  faces[i, 2], uvfaces[i, 2],
                #  faces[i, 1], uvfaces[i, 1],
                #  faces[i, 0], uvfaces[i, 0]
                faces[i, 0], uvfaces[i, 0],
                faces[i, 1], uvfaces[i, 1],
                faces[i, 2], uvfaces[i, 2]
            )
            )

# def faces_of_verts(all_faces, vert_idcs, consider_full_flame_model=False, return_face_idcs=False):
#     """
#     calculates face tensor of shape F x 3 with face spanned by vertices in flame mesh
#     all vertices of the faces returned by this function contain only vertices from vert_idcs
#     :param vert_idcs:
#     :param consider_full_flame_model:
#     :return_face_idcs: if True, also returns list of relevant face idcs
#     :return:
#     """
#     # all_faces = self._faces if consider_full_flame_model else self._faces_filtered  # F x 3
#     vert_idcs = vert_idcs.to(all_faces.device)
#     vert_faces = []
#     face_idcs = []
#     for i, f in enumerate(all_faces):
#         keep_face = True
#         for idx in f:
#             if not idx in vert_idcs:
#                 keep_face = False
#         if keep_face:
#             vert_faces.append(f)
#             face_idcs.append(i)
#     vert_faces = torch.stack(vert_faces)

#     if return_face_idcs:
#         return vert_faces, face_idcs

#     return vert_faces

def load_obj(path):
    """Load wavefront OBJ from file."""
    v = []
    vt = []
    vindices = []
    vtindices = []

    with open(path, "r") as f:
        while True:
            line = f.readline()

            if line == "":
                break

            if line[:2] == "v ":
                v.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "vt":
                vt.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "f ":
                vindices.append([int(entry.split('/')[0]) - 1 for entry in line.split()[1:]])
                if line.find("/") != -1:
                    vtindices.append([int(entry.split('/')[1]) - 1 for entry in line.split()[1:]])

    return v, vt, vindices, vtindices

def load_obj_multi_object(path):
    """Load wavefront OBJ from file."""
    obj_dict = {}

    with open(path, "r") as f:
        while True:
            line = f.readline()

            if line == "":
                break

            if line[:2] == "o ":
                name = line[2:].replace("\n", "")
                obj_dict[name] = []
            elif line[:2] == "v " or line[:2] == "vt" or line[:2] == "f ":
                obj_dict[name].append(line)

    obj_part = {}
    for part in obj_dict.keys():
        v = []
        vt = []
        vindices = []
        vtindices = []
        for line in obj_dict[part]:
            if line[:2] == "v ":
                v.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "vt":
                vt.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "f ":
                vindices.append([int(entry.split('/')[0]) - 1 for entry in line.split()[1:]])
                if line.find("/") != -1:
                    vtindices.append([int(entry.split('/')[1]) - 1 for entry in line.split()[1:]])
        obj_part[part] = [v, vt, vindices, vtindices]

    return obj_part

# def edge_subdivide(vertices, uvs, faces, uvfaces):
#     """
#     subdivides mesh based on edge midpoints. every triangle is subdivided into 4 child triangles.
#     old faces are kept in array
#     :param vertices: V x 3 ... vertex coordinates
#     :param uvs: T x 2 ... uv coordinates
#     :param faces: F x 3 face vertex idx array
#     :param uvfaces: F x 3 face uv idx array
#     :return:
#         - vertices ... np.array of vertex coordinates with shape V + n_edges x 3
#         - uvs ... np.array of uv coordinates with shape T + n_edges x 2
#         - faces ... np.array of face vertex idcs with shape F + 4*F x 3
#         - uv_faces ... np.array of face uv idcs with shape F + 4*F x 3
#         - edges ... np.array of shape n_edges x 2 giving the indices of the vertices of each edge
#         - uv_edges ... np.array of shape n_edges x 2 giving the indices of the uv_coords of each edge

#         all returns are a concatenation like np.concatenate((array_old, array_new), axis=0) so that
#         order of old entries is not changed and so that also old faces are still present.
#     """
#     n_faces = faces.shape[0]
#     n_vertices = vertices.shape[0]
#     n_uvs = uvs.shape[0]

#     # if self.edges is None:
#     # if True:
#     # compute edges
#     edges = []
#     edge_map = dict()
#     for i in range(0, n_faces):
#         append_edge(edge_map, edges, faces[i, 0], faces[i, 1])
#         append_edge(edge_map, edges, faces[i, 1], faces[i, 2])
#         append_edge(edge_map, edges, faces[i, 2], faces[i, 0])
#     n_edges = len(edges)
#     edges = np.array(edges).astype(int)

#     # compute edges uv space
#     uv_edges = []
#     uv_edge_map = dict()
#     for i in range(0, n_faces):
#         append_edge(uv_edge_map, uv_edges, uvfaces[i, 0], uvfaces[i, 1])
#         append_edge(uv_edge_map, uv_edges, uvfaces[i, 1], uvfaces[i, 2])
#         append_edge(uv_edge_map, uv_edges, uvfaces[i, 2], uvfaces[i, 0])
#     uv_n_edges = len(uv_edges)
#     uv_edges = np.array(uv_edges).astype(int)

#     #    print('edges:', edges.shape)
#     #    print('self.edge_map :', len(edge_map ))
#     #
#     #    print('uv_edges:', uv_edges.shape)
#     #    print('self.uv_edge_map :', len(uv_edge_map ))
#     #
#     #    print('vertices:', vertices.shape)
#     #    print('normals:', normals.shape)
#     #    print('uvs:', uvs.shape)
#     #    print('faces:', faces.shape)
#     #    print('uvfaces:', uvfaces.shape)

#     ############
#     # vertices
#     v = np.zeros((n_vertices + n_edges, 3))
#     # copy original vertices
#     v[:n_vertices, :] = vertices
#     # compute edge midpoints
#     vertices_edges = vertices[edges]
#     v[n_vertices:, :] = 0.5 * (vertices_edges[:, 0] + vertices_edges[:, 1])

#     # uvs
#     f_uvs = np.zeros((n_uvs + uv_n_edges, 2))
#     # copy original uvs
#     f_uvs[:n_uvs, :] = uvs
#     # compute edge midpoints
#     uvs_edges = uvs[uv_edges]
#     f_uvs[n_uvs:, :] = 0.5 * (uvs_edges[:, 0] + uvs_edges[:, 1])

#     # new topology
#     f = np.concatenate((faces, np.zeros((4 * n_faces, 3))), axis=0)
#     f_uv_id = np.concatenate((uvfaces, np.zeros((4 * n_faces, 3))), axis=0)
#     # f_uv = np.zeros((4*n_faces*3, 2))
#     for i in range(0, n_faces):
#         # vertex ids
#         a = int(faces[i, 0])
#         b = int(faces[i, 1])
#         c = int(faces[i, 2])
#         ab = n_vertices + edge_map[(a, b)]
#         bc = n_vertices + edge_map[(b, c)]
#         ca = n_vertices + edge_map[(c, a)]
#         # uvs
#         a_uv = int(uvfaces[i, 0])
#         b_uv = int(uvfaces[i, 1])
#         c_uv = int(uvfaces[i, 2])
#         ab_uv = n_uvs + uv_edge_map[(a_uv, b_uv)]
#         bc_uv = n_uvs + uv_edge_map[(b_uv, c_uv)]
#         ca_uv = n_uvs + uv_edge_map[(c_uv, a_uv)]

#         ## triangle 1
#         f[n_faces + 4 * i, 0] = a
#         f[n_faces + 4 * i, 1] = ab
#         f[n_faces + 4 * i, 2] = ca
#         f_uv_id[n_faces + 4 * i, 0] = a_uv
#         f_uv_id[n_faces + 4 * i, 1] = ab_uv
#         f_uv_id[n_faces + 4 * i, 2] = ca_uv

#         ## triangle 2
#         f[n_faces + 4 * i + 1, 0] = ab
#         f[n_faces + 4 * i + 1, 1] = b
#         f[n_faces + 4 * i + 1, 2] = bc
#         f_uv_id[n_faces + 4 * i + 1, 0] = ab_uv
#         f_uv_id[n_faces + 4 * i + 1, 1] = b_uv
#         f_uv_id[n_faces + 4 * i + 1, 2] = bc_uv

#         ## triangle 3
#         f[n_faces + 4 * i + 2, 0] = ca
#         f[n_faces + 4 * i + 2, 1] = ab
#         f[n_faces + 4 * i + 2, 2] = bc
#         f_uv_id[n_faces + 4 * i + 2, 0] = ca_uv
#         f_uv_id[n_faces + 4 * i + 2, 1] = ab_uv
#         f_uv_id[n_faces + 4 * i + 2, 2] = bc_uv

#         ## triangle 4
#         f[n_faces + 4 * i + 3, 0] = ca
#         f[n_faces + 4 * i + 3, 1] = bc
#         f[n_faces + 4 * i + 3, 2] = c
#         f_uv_id[n_faces + 4 * i + 3, 0] = ca_uv
#         f_uv_id[n_faces + 4 * i + 3, 1] = bc_uv
#         f_uv_id[n_faces + 4 * i + 3, 2] = c_uv

#     return v, f_uvs, f, f_uv_id

# def append_edge(edge_map_, edges_, idx_a, idx_b):
#     if idx_b < idx_a:
#         idx_a, idx_b = idx_b, idx_a

#     if not (idx_a, idx_b) in edge_map_:
#         e_id = len(edges_)
#         edges_.append([idx_a, idx_b])
#         edge_map_[(idx_a, idx_b)] = e_id
#         edge_map_[(idx_b, idx_a)] = e_id

def get_seg_mask(smplx_face, other_ids=None, layer_id=0, mask_ids=None):
    "following https://github.com/TingtingLiao/TADA/blob/main/lib/common/utils.py"

    smplx_segs = json.load(open(f'/home/zhangweitian/HighResAvatar/work_dirs/cache/template/smplx_vert_segmentation.json'))
    flame_segs = pickle.load(open(f'/home/zhangweitian/HighResAvatar/work_dirs/cache/template/FLAME_masks.pkl', 'rb'), encoding='latin1')
    smplx_flame_vid = np.load(f"/home/zhangweitian/HighResAvatar/work_dirs/cache/template/SMPL-X__FLAME_vertex_ids.npy", allow_pickle=True)
    # dict_keys(['rightHand', 'rightUpLeg', 'leftArm', 'head', 'leftEye', 'rightEye', 'leftLeg', 'leftToeBase', 'leftFoot', 'spine1', 'spine2', 'leftShoulder', 'rightShoulder', 'rightFoot', 'rightArm', 'leftHandIndex1', 'rightLeg', 'rightHandIndex1', 'leftForeArm', 'rightForeArm', 'neck', 'rightToeBase', 'spine', 'leftUpLeg', 'eyeballs', 'leftHand', 'hips'])

    # short_sleeve_cloth_long_pants_ids = smplx_segs["leftArm"] + smplx_segs["rightArm"] + smplx_segs["spine"] + smplx_segs["spine1"] + smplx_segs["spine2"] + smplx_segs["leftShoulder"] + \
    #                                     smplx_segs["rightShoulder"] + smplx_segs["leftUpLeg"] + smplx_segs["rightUpLeg"] + smplx_segs["leftLeg"] + smplx_segs["rightLeg"] + smplx_segs["hips"]

    eyeball_ids = smplx_segs["leftEye"] + smplx_segs["rightEye"]
    hands_ids = smplx_segs["leftHand"] + smplx_segs["rightHand"] + \
                smplx_segs["leftHandIndex1"] + smplx_segs["rightHandIndex1"]
    feet_ids = smplx_segs['leftFoot'] + smplx_segs['rightFoot'] + smplx_segs['leftToeBase'] + smplx_segs['rightToeBase'] 

    front_face_ids = list(smplx_flame_vid[flame_segs["face"]])
    ears_ids = list(smplx_flame_vid[flame_segs["left_ear"]]) + list(smplx_flame_vid[flame_segs["right_ear"]])
    forehead_ids = list(smplx_flame_vid[flame_segs["forehead"]])

    label_flame = []
    for key in flame_segs:
        label_flame += list(smplx_flame_vid[flame_segs[key]])
    outside_ids = list(set(smplx_flame_vid) - set(label_flame))
    # mouth_inner
    lips_ids = list(smplx_flame_vid[flame_segs["lips"]])
    nose_ids = list(smplx_flame_vid[flame_segs["nose"]])
    scalp_ids = list(smplx_flame_vid[flame_segs["scalp"]])
    hair_ids = list(set(scalp_ids + forehead_ids)) + smplx_segs["spine"] + smplx_segs["spine1"] + smplx_segs["spine2"] + smplx_segs["neck"]
    shoes_ids = feet_ids + smplx_segs["leftLeg"] + smplx_segs["rightLeg"]
    up_ids = smplx_segs["leftArm"] + smplx_segs["rightArm"] + smplx_segs["spine"] + smplx_segs["spine1"] + smplx_segs["spine2"] + smplx_segs["leftShoulder"] + \
            smplx_segs["rightShoulder"] + smplx_segs["hips"] + smplx_segs["leftForeArm"] + smplx_segs["rightForeArm"]
    lower_ids = smplx_segs["leftUpLeg"] + smplx_segs["rightUpLeg"] + smplx_segs["leftLeg"] + smplx_segs["rightLeg"] + smplx_segs["hips"] 

    if layer_id == 0:
        if other_ids != None:
            remesh_ids = list(set(front_face_ids) - set(forehead_ids) - set(other_ids)) + ears_ids + eyeball_ids + hands_ids + smplx_segs["spine"] + smplx_segs["spine1"] + smplx_segs["spine2"] + smplx_segs["hips"] + feet_ids + scalp_ids
        else:
            remesh_ids = list(set(front_face_ids) - set(forehead_ids)) + ears_ids + eyeball_ids + hands_ids
        remesh_mask = ~np.isin(np.arange(10475), remesh_ids) # obtain selected vertices
    elif layer_id == 1 or layer_id == 2:
        remesh_ids = mask_ids[0, :, 0].tolist()
        remesh_mask = np.isin(np.arange(10475), remesh_ids) # obtain selected vertices
    else:
        assert False
    # remesh_mask = ~np.isin(np.arange(10475), remesh_ids) # obtain selected vertices
    remesh_mask = remesh_mask[smplx_face].all(axis=1) # obtain selected mesh

    # offset_ids = list(set(lips_ids) - set(other_ids)) + hands_ids
    face_ids = list(set(front_face_ids) - set(forehead_ids)) + eyeball_ids
    face_mask = ~np.isin(np.arange(10475), face_ids)
    face_mask = face_mask[smplx_face].all(axis=1)

    hands_mask = ~np.isin(np.arange(10475), hands_ids)
    hands_mask = hands_mask[smplx_face].all(axis=1)

    # feet_mask = ~np.isin(np.arange(10475), feet_ids)
    # feet_mask = feet_mask[smplx_face].all(axis=1)

    # ear_mask = ~np.isin(np.arange(10475), ears_ids)
    # ear_mask = ear_mask[smplx_face].all(axis=1)

    outside_mask = ~np.isin(np.arange(10475), outside_ids)
    outside_mask = outside_mask[smplx_face].all(axis=1)

    # mouth_in_mask = ~np.isin(np.arange(10475), other_ids)
    # mouth_in_mask = mouth_in_mask[smplx_face].all(axis=1)

    nose_mask =  ~np.isin(np.arange(10475), nose_ids)
    nose_mask =  nose_mask[smplx_face].all(axis=1)

    # lip_mask =  ~np.isin(np.arange(10475), lips_ids)
    # lip_mask =  lip_mask[smplx_face].all(axis=1)

    # cloth_template_mask =  ~np.isin(np.arange(10475), short_sleeve_cloth_long_pants_ids)
    # cloth_template_mask =  cloth_template_mask[smplx_face].all(axis=1)

    # for layer 1 so it's different from normal mask version
    if layer_id == 1:
        hair_mask =  ~np.isin(remesh_ids, hair_ids)
        shoes_mask =  ~np.isin(remesh_ids, shoes_ids)
    elif layer_id == 2:
        up_mask =  ~np.isin(remesh_ids, up_ids)
        low_mask =  ~np.isin(remesh_ids, lower_ids)
        hip_mask = np.isin(remesh_ids, smplx_segs["hips"])
        hip_remesh_ids = [i for i in range(hip_mask.shape[0]) if hip_mask[i] == 1]
        print(hip_remesh_ids[540:550])
        print(hip_remesh_ids[550:560])
        assert False

    # whole_face_mask = (~outside_mask + ~face_mask)
    # whole_hands_mask = ~hands_mask
    # print(whole_face_mask.shape)
    # print(whole_face_mask.sum())
    # np.save('/home/zhangweitian/HighResAvatar/debug/face_mask.npy', np.array(whole_face_mask))
    # np.save('/home/zhangweitian/HighResAvatar/debug/hands_mask.npy', np.array(whole_hands_mask))
    # assert False


    # print(len(offset_mask))
    # offset_mask = lips_ids + hands_ids
    # print(len(offset_mask))
    # assert False
    # return remesh_mask, face_mask, hands_mask, outside_mask, hair_mask, shoes_mask
    return remesh_mask, face_mask, hands_mask, outside_mask, up_mask, low_mask

def auto_uv(v, f):
    v_np = v
    f_np = f
    
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)
    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 4
    atlas.generate(chart_options=chart_options)
    vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

    return vt_np, ft_np


if __name__ == '__main__':
    #### Obtain subdivide regions
    smplx_mesh = load_obj('/home/zhangweitian/HighResAvatar/work_dirs/cache/template/body.obj')
    # hair_and_shoes_mesh = load_obj('/home/zhangweitian/HighResAvatar/work_dirs/cache/template/hair_shoes.obj')
    # up_and_lower_mesh = load_obj_multi_object('/home/zhangweitian/HighResAvatar/work_dirs/cache/template/up_and_low_new.obj')
    # coat_mesh = load_obj('/home/zhangweitian/HighResAvatar/work_dirs/cache/template/coat_part.obj')
    # low_verts = np.array(up_and_lower_mesh['low'][0]).astype(np.float32)
    # low_uv = np.array(up_and_lower_mesh['low'][1])
    # low_faces = np.array(up_and_lower_mesh['low'][2])
    # low_uv_faces = np.array(up_and_lower_mesh['low'][3]) 
    # up_verts = np.array(up_and_lower_mesh['up'][0]).astype(np.float32)
    # up_uv = np.array(up_and_lower_mesh['up'][1])
    # up_faces = np.array(up_and_lower_mesh['up'][2])
    # up_uv_faces = np.array(up_and_lower_mesh['up'][3]) 
    # layer_2_verts = np.concatenate([low_verts, up_verts], axis=0)
    # layer_2_uv = np.concatenate([low_uv, up_uv], axis=0)
    # layer_2_faces = np.concatenate([low_faces, up_faces], axis=0)
    # layer_2_uv_faces = np.concatenate([low_uv_faces, up_uv_faces], axis=0)
    # layer_1_verts = np.array(hair_and_shoes_mesh[0]).astype(np.float32)
    # layer_1_uv = np.array(hair_and_shoes_mesh[1])
    # layer_1_faces = np.array(hair_and_shoes_mesh[2])
    # layer_1_uv_faces = np.array(hair_and_shoes_mesh[3]) 
    smplx_verts = smplx_mesh[0]
    smplx_uv = np.array(smplx_mesh[1])
    smplx_faces = np.array(smplx_mesh[2])
    mouth_inner_faces = smplx_faces[-30:]
    mouth_inner_ids = list(set(mouth_inner_faces.flatten().tolist()))
    smplx_uv_faces = np.array(smplx_mesh[3]) 
    # subdivide for layer 0
    # remesh_mask, face_mask, hands_mask, outside_mask, nose_mask = get_seg_mask(smplx_faces, mouth_inner_ids, layer_id=0)
    # subdivide for layer 1
    # dist, layer_1_ids, _ = knn_points(torch.tensor(layer_1_verts).unsqueeze(0), torch.tensor(smplx_verts).unsqueeze(0))
    # layer_1_ids = np.array(layer_1_ids)
    # remesh_mask, face_mask, hands_mask, outside_mask, hair_mask, shoes_mask = get_seg_mask(smplx_faces, mouth_inner_ids, layer_id=1, mask_ids=layer_1_ids)
    # hair_mask =  hair_mask[layer_1_faces].all(axis=1)
    # shoes_mask = shoes_mask[layer_1_faces].all(axis=1)
    # print(layer_1_faces.shape)
    # print(hair_mask.sum())
    # print(shoes_mask.sum())
    # assert False
    # subdivide for layer 2
    # dist, layer_2_ids, _ = knn_points(torch.tensor(layer_2_verts).unsqueeze(0), torch.tensor(smplx_verts).unsqueeze(0))
    # layer_2_ids = np.array(layer_2_ids)
    # low_mask = np.zeros_like(layer_2_faces)[..., 0]
    # low_mask[:low_faces.shape[0]] = 1
    # low_mask = low_mask.astype(np.bool_)
    # up_mask = np.zeros_like(layer_2_faces)[..., 0]
    # up_mask[-up_faces.shape[0]:] = 1
    # up_mask = up_mask.astype(np.bool_)
    
    body_model = smplx.SMPLX('/home/zhangweitian/HighResAvatar/lib/models/deformers/smplx/SMPLX', gender='neutral', \
                            use_pca=False,
                            num_betas=10,
                            flat_hand_mean=True)
    body_pose_t = torch.zeros((1, 63))
    # body_pose_t[0, 2] = np.pi / 18
    # body_pose_t[0, 5] = -np.pi / 18

    # For custom
    # jaw_pose_t = torch.as_tensor([[0.2, 0, 0],])
    # transl = torch.as_tensor([[-0.0012,  0.4668, -0.0127],])

    # For THuman
    jaw_pose_t = torch.as_tensor([[0., 0, 0],])
    transl = torch.as_tensor([[0., 0.35, 0.],])
    left_hand_pose_t = torch.zeros((1, 45))
    right_hand_pose_t = torch.zeros((1, 45))
    # left_hand_pose_t = torch.tensor([[1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
    #      -0.6652, -0.7290,  0.0084, -0.4818],])
    # right_hand_pose_t = torch.tensor([[1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
    #      -0.6652, -0.7290,  0.0084, -0.4818],])
    
    global_orient = torch.zeros((1, 3))
    betas = torch.zeros((1, 10))
    # For custom
    # smpl_outputs = body_model(betas=betas, body_pose=body_pose_t, transl=transl, global_orient = global_orient, jaw_pose=jaw_pose_t)
    # For THuman
    smpl_outputs = body_model(betas=betas, body_pose=body_pose_t, transl=transl, global_orient = global_orient, jaw_pose=jaw_pose_t, left_hand_pose=left_hand_pose_t, right_hand_pose=right_hand_pose_t)
    vs_template = smpl_outputs.vertices.detach()[0]

    #### subdivide process
    num_remeshing = 1
    shapedirs = torch.cat([body_model.shapedirs, body_model.expr_dirs], dim=-1).reshape(10475, -1)
    posedirs = body_model.posedirs.reshape(-1, 10475, 3).permute(1, 2, 0)
    
    blending_weights = body_model.lbs_weights
    # layer 0
    # dense_v_cano, dense_faces, dense_blendweights, _ = subdivide(vs_template.clone(), smplx_faces[remesh_mask].copy(), blending_weights.clone())
    # _, _, dense_shapedirs, _ = subdivide(vs_template.clone(), smplx_faces[remesh_mask].copy(), shapedirs.clone())
    # _, _, dense_posedirs, _ = subdivide(vs_template.clone(), smplx_faces[remesh_mask].copy(), posedirs.clone())
    # dense_uv, dense_uv_faces, _ = subdivide(smplx_uv, smplx_uv_faces[remesh_mask])
    # np.save('../../../work_dirs/cache/body_podir_smplx_cop.npy', dense_posedirs.transpose(2, 0, 1).reshape(486, -1))
    # np.save('../../../work_dirs/cache/body_spdir_smplx_cop.npy', dense_shapedirs.reshape(-1, 3, 20))
    # layer 1
    # dense_v_cano_layer1, dense_faces_layer1, dense_blendweights_layer1, _ = subdivide(layer_1_verts.copy(), layer_1_faces.copy(), blending_weights[layer_1_ids[0, :, 0]].clone())
    # _, _, dense_shapedirs_l1, _ = subdivide(layer_1_verts.copy(), layer_1_faces.copy(), shapedirs[layer_1_ids[0, :, 0]].clone())
    # _, _, dense_posedirs_l1, _ = subdivide(layer_1_verts.copy(), layer_1_faces.copy(), posedirs[layer_1_ids[0, :, 0]].clone())
    # dense_uv_layer1, dense_uv_faces_layer1, _ = subdivide(layer_1_uv, layer_1_uv_faces)
    # dense_hair_mask = hair_mask.repeat(4)
    # dense_shoes_mask = shoes_mask.repeat(4)
    # np.save('../../../work_dirs/cache/hair_mask_l1.npy', np.array(~dense_hair_mask))
    # np.save('../../../work_dirs/cache/shoes_mask_l1.npy', np.array(~dense_shoes_mask))
    # np.save('../../../work_dirs/cache/hair_shoe_podir_smplx_cop.npy', dense_posedirs_l1.transpose(2, 0, 1).reshape(486, -1))
    # np.save('../../../work_dirs/cache/hair_shoe_spdir_smplx_cop.npy', dense_shapedirs_l1.reshape(-1, 3, 20))
    # layer 2
    dense_v_cano_layer2, dense_faces_layer2, dense_blendweights_layer2, _ = subdivide(layer_2_verts.copy(), layer_2_faces.copy(), blending_weights[layer_2_ids[0, :, 0]].clone())
    _, _, dense_shapedirs_l2, _ = subdivide(layer_2_verts.copy(), layer_2_faces.copy(), shapedirs[layer_2_ids[0, :, 0]].clone())
    _, _, dense_posedirs_l2, _ = subdivide(layer_2_verts.copy(), layer_2_faces.copy(), posedirs[layer_2_ids[0, :, 0]].clone())
    dense_uv_layer2, dense_uv_faces_layer2, _ = subdivide(layer_2_uv, layer_2_uv_faces)
    dense_up_mask = up_mask.repeat(4)
    dense_low_mask = low_mask.repeat(4)
    np.save('../../../work_dirs/cache/up_mask_l2.npy', np.array(dense_up_mask))
    np.save('../../../work_dirs/cache/low_mask_l2.npy', np.array(dense_low_mask))
    np.save('../../../work_dirs/cache/uplow_podir_smplx_cop.npy', dense_posedirs_l2.transpose(2, 0, 1).reshape(486, -1))
    np.save('../../../work_dirs/cache/uplow_spdir_smplx_cop.npy', dense_shapedirs_l2.reshape(-1, 3, 20))
    # assert False


    # layer 0
    # mask_part1 = outside_mask[remesh_mask]
    # dense_outside_mask = mask_part1.repeat(4)

    # mask_face_part = face_mask[remesh_mask]
    # dense_face_mask = mask_face_part.repeat(4)

    # mask_nose_part = nose_mask[remesh_mask]
    # dense_nose_mask = mask_nose_part.repeat(4)

    # dense_face_masks = np.concatenate([dense_face_mask, face_mask[~remesh_mask]])
    # dense_nose_masks = np.concatenate([dense_nose_mask, nose_mask[~remesh_mask]])
    # dense_hands_masks = np.concatenate([np.ones(dense_faces.shape[0]), hands_mask[~remesh_mask]])
    # dense_outside_masks = np.concatenate([dense_outside_mask, outside_mask[~remesh_mask]])

    # dense_face_masks = torch.as_tensor(dense_face_masks).bool()
    # dense_hands_masks = torch.as_tensor(dense_hands_masks).bool()
    # dense_outside_masks = torch.as_tensor(dense_outside_masks).bool()
    # dense_nose_masks = torch.as_tensor(dense_nose_masks).bool()
    
    # np.save('../../../work_dirs/cache/nose_mask_body.npy', np.array(~dense_nose_masks))
    # assert False
    # np.save('../../../work_dirs/cache/outside_mask_body.npy', np.array(~dense_outside_masks))
    # np.save('../../../work_dirs/cache/face_mask_body.npy', np.array(~dense_face_masks))
    # np.save('../../../work_dirs/cache/hands_mask_body.npy', np.array(~dense_hands_masks))

    # dense_faces = np.concatenate([dense_faces, smplx_faces[~remesh_mask]])
    # dense_uv_faces = np.concatenate([dense_uv_faces, smplx_uv_faces[~remesh_mask]])
    # final_blendweights = np.stack([dense_blendweights[dense_faces[..., i]] for i in range(3)], axis=1).mean(1)
    # np.save('../../../work_dirs/cache/init_lbsw_smplx_body.npy', final_blendweights)

    # for _ in range(1, num_remeshing):
    #     assert False
    #     dense_v_cano, dense_faces, _ = subdivide(dense_v_cano, dense_faces)
    #     dense_uv, dense_uv_faces, _ = subdivide(dense_uv, dense_uv_faces)

    #### export densified mesh
    # layer 0
    # print(dense_faces.shape)
    # write_obj('/home/zhangweitian/HighResAvatar/work_dirs/cache/template/dense_body.obj', dense_v_cano, dense_faces, dense_uv, dense_uv_faces)
    # layer 1
    # print(dense_faces_layer1.shape)
    # write_obj('/home/zhangweitian/HighResAvatar/work_dirs/cache/template/dense_hair_shoes.obj', dense_v_cano_layer1, dense_faces_layer1, dense_uv_layer1, dense_uv_faces_layer1)
    # layer 2
    print(dense_faces_layer2.shape)
    write_obj('/home/zhangweitian/HighResAvatar/work_dirs/cache/template/dense_up_and_low.obj', dense_v_cano_layer2, dense_faces_layer2, dense_uv_layer2, dense_uv_faces_layer2)

