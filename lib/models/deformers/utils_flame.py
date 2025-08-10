import numpy as np
import torch
import torch.nn.functional as F
from flame.flame_deca.FLAME import FLAME
import trimesh
import os


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

def gentritex(v, vt, vi, vti, texsize):
    """Create 3 texture maps containing the vertex indices, texture vertex
    indices, and barycentric coordinates"""
    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)
    ntris = vi.shape[0]

    texu, texv = np.meshgrid(
            (np.arange(texsize) + 0.5) / texsize,
            (np.arange(texsize) + 0.5) / texsize)
    texuv = np.stack((texu, texv), axis=-1)

    vt = vt[vti]

    viim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    vtiim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    baryim = np.zeros((texsize, texsize, 3), dtype=np.float32)

    for i in list(range(ntris))[::-1]:
        bbox = (
            max(0, int(min(vt[i, 0, 0], min(vt[i, 1, 0], vt[i, 2, 0]))*texsize)-1),
            min(texsize, int(max(vt[i, 0, 0], max(vt[i, 1, 0], vt[i, 2, 0]))*texsize)+2),
            max(0, int(min(vt[i, 0, 1], min(vt[i, 1, 1], vt[i, 2, 1]))*texsize)-1),
            min(texsize, int(max(vt[i, 0, 1], max(vt[i, 1, 1], vt[i, 2, 1]))*texsize)+2))
        v0 = vt[None, None, i, 1, :] - vt[None, None, i, 0, :]
        v1 = vt[None, None, i, 2, :] - vt[None, None, i, 0, :]
        v2 = texuv[bbox[2]:bbox[3], bbox[0]:bbox[1], :] - vt[None, None, i, 0, :]
        d00 = np.sum(v0 * v0, axis=-1)
        d01 = np.sum(v0 * v1, axis=-1)
        d11 = np.sum(v1 * v1, axis=-1)
        d20 = np.sum(v2 * v0, axis=-1)
        d21 = np.sum(v2 * v1, axis=-1)
        denom = d00 * d11 - d01 * d01

        if denom != 0.:
            baryv = (d11 * d20 - d01 * d21) / denom
            baryw = (d00 * d21 - d01 * d20) / denom
            baryu = 1. - baryv - baryw

            baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((baryu, baryv, baryw), axis=-1),
                    baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((vi[i, 0], vi[i, 1], vi[i, 2]), axis=-1),
                    viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((vti[i, 0], vti[i, 1], vti[i, 2]), axis=-1),
                    vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])

    return viim, vtiim, baryim

@torch.jit.script
def compute_tbn(v0, v1, v2, vt0, vt1, vt2):
    v01 = v1 - v0
    v02 = v2 - v0
    vt01 = vt1 - vt0
    vt02 = vt2 - vt0
    f = 1. / (vt01[None, :, :, 0] * vt02[None, :, :, 1] - vt01[None, :, :, 1] * vt02[None, :, :, 0])
    # print(torch.isinf(f).sum())
    # assert False
    tangent = f[:, :, :, None] * torch.stack([
        v01[:, :, :, 0] * vt02[None, :, :, 1] - v02[:, :, :, 0] * vt01[None, :, :, 1],
        v01[:, :, :, 1] * vt02[None, :, :, 1] - v02[:, :, :, 1] * vt01[None, :, :, 1],
        v01[:, :, :, 2] * vt02[None, :, :, 1] - v02[:, :, :, 2] * vt01[None, :, :, 1]], dim=-1)
    # print(tangent[0, 0, 0])
    # assert False
    tangent = F.normalize(tangent, dim=-1)
    normal = torch.cross(v01, v02, dim=3)
    normal = F.normalize(normal, dim=-1)
    bitangent = torch.cross(tangent, normal, dim=3)
    bitangent = F.normalize(bitangent, dim=-1)

    # create matrix
    primrotmesh = torch.stack((tangent, bitangent, normal), dim=-1)
    # print(primrotmesh[0, 0, 0])
    # assert False

    return primrotmesh


if __name__ == '__main__':
    flame_model = FLAME(flame_model_path='/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/flame/flame_model/generic_model.pkl',
                           flame_lmk_embedding_path='/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/flame/flame_model/landmark_embedding.npy')
    # body_model = SMPL('/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/flame/flame_model/generic_model.pkl', gender='neutral', \
    #                             create_body_pose=False, \
    #                             create_betas=False, \
    #                             create_global_orient=False, \
    #                             create_transl=False)

    flame_param_t = torch.zeros((1, 169))
    flame_param_t[0, 0] = 1
    flame_param_t[0, 10] = 0.2 # openmouth slightly
    flame_outputs = flame_model(shape_params=flame_param_t[:, 19:119], expression_params=flame_param_t[:, 119:], pose_params=flame_param_t[:, 4:19])
    vs_template = flame_outputs.vertices
    
    # final_mesh = trimesh.Trimesh(vertices=vs_template[0].cpu().detach().numpy(), faces=flame_outputs.faces.cpu().detach().numpy())
    # final_mesh.export(os.path.join('/mnt/sdb/zwt/SSDNeRF/debug', 'canonical_head' + '.obj'))
    # assert False
    # new_verts, new_faces = trimesh.remesh.subdivide(vs_template[0].cpu().detach().numpy(), flame_outputs.faces.cpu().detach().numpy())
    # new_verts, new_faces = trimesh.remesh.subdivide(new_verts, new_faces)
    # print(new_verts.shape)
    # final_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
    # final_mesh.export(os.path.join('/mnt/sdb/zwt/SSDNeRF/debug', 'canonical_head' + '.obj'))
    # assert False
    # select_idx = np.random.choice(new_verts.shape[0], 55000, replace=False)
    # select_pcd = new_verts[select_idx]
    # mask = select_pcd[:, 1] > -1.0
    # mask = final_pos[:, 1] < -0.75
    # final_pos = select_pcd[mask]
    # final_rot = torch.eye(3).unsqueeze(0).repeat(final_pos.shape[0], 1, 1)
    # print(final_pos.shape)
    # np.save('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_pcd_flame.npy', final_pos)
    # np.save('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_rot_flame.npy', final_rot)
    # assert False
    

    objpath = "/mnt/sdb/zwt/SSDNeRF/debug/head_template.obj" # from DECA repo
    v, vt, vi, vti = load_obj(objpath)
    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)
    
    idxim, tidxim, barim = gentritex(v, vt, vi, vti, 256)
    # v = torch.as_tensor(np.array(v, dtype=np.float32).reshape(1, -1, 3))
    idxim = torch.tensor(idxim).long()
    tidxim = torch.tensor(tidxim).long()
    barim = torch.tensor(barim)
    vt = torch.as_tensor(vt)
    
    uvheight, uvwidth = barim.size(0), barim.size(1)
    stridey = uvheight // uvheight
    stridex = uvwidth // uvwidth

    # print(len(v))
    # print(vs_template.shape)
    # assert False
    v = vs_template
    # v = torch.as_tensor(v).unsqueeze(0)
    v0 = v[:, idxim[stridey//2::stridey, stridex//2::stridex, 0], :]
    v1 = v[:, idxim[stridey//2::stridey, stridex//2::stridex, 1], :]
    v2 = v[:, idxim[stridey//2::stridey, stridex//2::stridex, 2], :]

    vt0 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 0], :]
    vt1 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 1], :]
    vt2 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 2], :]

    primrotmesh = compute_tbn(v0, v1, v2, vt0, vt1, vt2).view(v0.size(0), -1, 3, 3)
    # print(primrotmesh[:3])

    primposmesh = (
                    barim[None, stridey//2::stridey, stridex//2::stridex, 0, None] * v0 +
                    barim[None, stridey//2::stridey, stridex//2::stridex, 1, None] * v1 +
                    barim[None, stridey//2::stridey, stridex//2::stridex, 2, None] * v2
                    ).view(v0.size(0), -1, 3)
    
    primuv = (
                    barim[None, stridey//2::stridey, stridex//2::stridex, 0, None] * vt0 +
                    barim[None, stridey//2::stridey, stridex//2::stridex, 1, None] * vt1 +
                    barim[None, stridey//2::stridey, stridex//2::stridex, 2, None] * vt2
                    ).view(v0.size(0), -1, 2)
    
    test_pos = primposmesh[0].numpy().tolist()
    test_rot = primrotmesh[0].numpy().tolist()
    test_uv = primuv[0].numpy().tolist()
    final_pos = []
    final_idx = torch.zeros(primposmesh.shape[1]).bool()
    for idx, pos in enumerate(test_pos):
        if pos not in final_pos:
            final_pos.append(pos)
            final_idx[idx] = True

    final_pos = torch.as_tensor(final_pos)[1:]
    final_rot = torch.as_tensor(test_rot)[final_idx][1:]
    final_uv = torch.as_tensor(test_uv)[final_idx][1:]
    mask = final_pos[:, 1] > -1.0
    # mask = final_pos[:, 1] < -0.75
    final_pos = final_pos[mask]
    final_rot = final_rot[mask]
    final_uv = final_uv[mask]
    print(final_pos.shape)
    print(final_rot.shape)
    print(final_uv.shape)
    # boundary_pos = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_pcd_boundary.npy'))
    # boundary_rot = torch.eye(3).unsqueeze(0).repeat(boundary_pos.shape[0], 1, 1)
    # final_pos = torch.cat([final_pos, boundary_pos], dim=0)
    # final_rot = torch.cat([final_rot, boundary_rot], dim=0)
    # print(final_pos.shape)
    # print(final_rot.shape)
    # np.save('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_uv_flame.npy', final_uv.numpy())
    np.save('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_pcd_flame.npy', final_pos.numpy())
    np.save('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_rot_flame.npy', final_rot.numpy())