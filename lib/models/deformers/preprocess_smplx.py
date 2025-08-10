import trimesh
import numpy as np
import os
from smplx import SMPLX
import torch


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
        # write mtl
        # with open(mtl_name, 'w') as f:
        #     f.write('newmtl %s\n' % material_name)
        #     s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
        #     f.write(s)

        #     if normal_map is not None:
        #         name, _ = os.path.splitext(obj_name)
        #         normal_name = f'{name}_normals.png'
        #         f.write(f'disp {normal_name}')
        #         # out_normal_map = normal_map / (np.linalg.norm(
        #         #     normal_map, axis=-1, keepdims=True) + 1e-9)
        #         # out_normal_map = (out_normal_map + 1) * 0.5

        #         cv2.imwrite(
        #             normal_name,
        #             # (out_normal_map * 255).astype(np.uint8)[:, :, ::-1]
        #             normal_map
        #         )
        # cv2.imwrite(texture_name, texture)


if __name__ == '__main__':
    # body_model = SMPLX('/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/smplx/SMPLX', gender='male', \
    #                             use_pca=True,
    #                             num_pca_comps=12,
    #                             num_betas=10,
    #                             flat_hand_mean=True)
    # body_pose_t = torch.zeros((1, 63))
    # transl = torch.as_tensor([[-0.0012,  0.4668, -0.0127],])
    # global_orient = torch.zeros((1, 3))
    # betas = torch.zeros((1, 10))
    # jaw_pose = torch.as_tensor([[0.2, 0, 0],])
    # smpl_outputs = body_model(betas=betas, body_pose=body_pose_t, transl=transl, global_orient = global_orient, jaw_pose=jaw_pose)
    # vs_template = smpl_outputs.vertices.detach()

    # v, vt, vi, vti
    flame_mouth_mesh = load_obj('/home/zhangweitian/HighResAvatar/work_dirs/cache/template/head_template_mesh_mouth.obj')
    flame_mouth_faces = flame_mouth_mesh[2]
    flame_mouth_uv = flame_mouth_mesh[1]
    flame_mouth_uv_faces = flame_mouth_mesh[3]
    flame_mesh = load_obj('/home/zhangweitian/HighResAvatar/work_dirs/cache/template/head_template.obj')
    flame_faces = flame_mesh[2]
    flame_uv = flame_mesh[1]
    flame_uv_faces = flame_mesh[3]
    # smplx_mesh = load_obj('/mnt/sdb/zwt/SSDNeRF/debug/template_mesh_smplx_uv.obj')
    smplx_mesh = load_obj('/home/zhangweitian/HighResAvatar/work_dirs/cache/template/smplx_uv.obj')
    smplx_verts = smplx_mesh[0]
    # smplx_verts = vs_template[0]
    smplx_uv = smplx_mesh[1]
    smplx_faces = smplx_mesh[2]
    smplx_uv_faces = smplx_mesh[3]

    extra_uvs = [uv for uv in flame_mouth_uv if uv not in flame_uv]
    extra_uvs = np.array(extra_uvs)
    extra_uvs *= np.array([[0.1, 0.06], ])
    # extra_uvs = extra_uvs * 0.06
    extra_uvs += np.array([[0.52, 0.55], ])
    # extra_uvs_idx = [i for i, uv in enumerate(flame_mouth_uv) if uv not in flame_uv]
    extra_faces = [face for face in flame_mouth_faces if face not in flame_faces]
    extra_uv_faces = [face for face in flame_mouth_uv_faces if face not in flame_uv_faces]
    extra_uv_faces_smplx = np.array(extra_uv_faces) - len(flame_uv) + len(smplx_uv)
    # extra_uvs = [uv for uv in flame_mouth_uv if uv not in flame_uv]
    # print(len(extra_faces))
    # print(len(extra_uvs))
    # print(extra_uvs)
    # smplx_mesh = load_obj('/mnt/sdb/zwt/SSDNeRF/debug/template_mesh_smplx_uv.obj')
    # print(np.array(smplx_mesh[0]).shape)
    smplx_flame_correspond = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/template/SMPL-X__FLAME_vertex_ids.npy')
    # smplx_flame_uv_correspond = 
    extra_faces_smplx = []
    for face in extra_faces:
        new_faces = [smplx_flame_correspond[idx] for idx in face]
        extra_faces_smplx.append(new_faces)

    # write obj
    vertices = np.array(smplx_verts)
    uvcoords = np.array(smplx_uv + extra_uvs.tolist())
    faces = np.array(smplx_faces + extra_faces_smplx)
    print(len(extra_faces_smplx))
    assert False
    uvfaces = np.array(smplx_uv_faces + extra_uv_faces_smplx.tolist())
    output_obj_path = '/home/zhangweitian/HighResAvatar/work_dirs/cache/template/smplx_mouth_uv.obj'
    write_obj(output_obj_path, vertices, faces, uvcoords=uvcoords, uvfaces=uvfaces)
    
    

