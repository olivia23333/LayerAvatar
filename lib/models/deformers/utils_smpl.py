import numpy as np
import torch
import torch.nn.functional as F
from smplx import SMPL


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
    body_model = SMPL('/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/smplx/SMPLX', gender='neutral', \
                                create_body_pose=False, \
                                create_betas=False, \
                                create_global_orient=False, \
                                create_transl=False)
    body_pose_t = torch.zeros((1, 69))
    transl = torch.zeros((1, 3))
    transl[:,1] = 0.3
    global_orient = torch.zeros((1, 3))
    betas = torch.zeros((1, 10))
    smpl_outputs = body_model(betas=betas, body_pose=body_pose_t, transl=transl, global_orient = global_orient)
    vs_template = smpl_outputs.vertices

    objpath = "/mnt/sdb/zwt/SSDNeRF/debug/template_mesh_smpl_uv.obj"
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

    v = vs_template
    v0 = v[:, idxim[stridey//2::stridey, stridex//2::stridex, 0], :]
    v1 = v[:, idxim[stridey//2::stridey, stridex//2::stridex, 1], :]
    v2 = v[:, idxim[stridey//2::stridey, stridex//2::stridex, 2], :]

    vt0 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 0], :]
    vt1 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 1], :]
    vt2 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 2], :]

    # primrotmesh = compute_tbn(v0, v1, v2, vt0, vt1, vt2).view(v0.size(0), -1, 3, 3)
    primrotmesh = compute_tbn(v0, v1, v2, vt0, vt1, vt2)
    # print(primrotmesh[:3])

    primposmesh = (
                    barim[None, stridey//2::stridey, stridex//2::stridex, 0, None] * v0 +
                    barim[None, stridey//2::stridey, stridex//2::stridex, 1, None] * v1 +
                    barim[None, stridey//2::stridey, stridex//2::stridex, 2, None] * v2
                    )
    # primposmesh = (
    #                 barim[None, stridey//2::stridey, stridex//2::stridex, 0, None] * v0 +
    #                 barim[None, stridey//2::stridey, stridex//2::stridex, 1, None] * v1 +
    #                 barim[None, stridey//2::stridey, stridex//2::stridex, 2, None] * v2
    #                 ).view(v0.size(0), -1, 3)
    primposmesh_vis_mask = (primposmesh==0)
    primnormesh = primrotmesh[..., -1]
    primnormesh_vis = torch.round((primnormesh * 0.5 + 0.5)*255).numpy()
    # primposmesh_vis = torch.round((primposmesh * 0.5 + 0.5)*255).numpy()
    primposmesh_vis_mask = torch.round((~primposmesh_vis_mask).float()*255).numpy()
    import cv2
    # cv2.imwrite('/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/uv_pos.png', primposmesh_vis[0])
    # print(np.concatenate([primposmesh_vis[0], primposmesh_vis_mask[0, :,:,0:1]], axis=2).shape)
    # assert False
    cv2.imwrite('/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/uv_norm.png', np.concatenate([primnormesh_vis[0], primposmesh_vis_mask[0, :,:,0:1]], axis=2))
    assert False
    
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

    final_pos = torch.as_tensor(final_pos)
    print(final_pos.min(0))
    print(final_pos.max(0))
    final_rot = torch.as_tensor(test_rot)[final_idx]
    print(final_pos[1:].shape)
    print(final_rot[1:].shape)
    final_uv = torch.as_tensor(test_uv)[final_idx]
    print(final_uv[1:].shape)
    final_idxim = idxim.reshape(-1, 3)[final_idx]
    final_barim = barim.reshape(-1, 3)[final_idx]
    print(final_barim.shape)
    print(final_idxim.shape)
    np.save('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_idx_smpl.npy', final_idxim[1:].numpy())
    np.save('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_bar_smpl.npy', final_barim[1:].numpy())
    assert False
    np.save('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_uv_smpl.npy', final_uv[1:].numpy())
    assert False
    np.save('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_pcd_smpl.npy', final_pos[1:].numpy())
    np.save('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_rot_smpl.npy', final_rot[1:].numpy())