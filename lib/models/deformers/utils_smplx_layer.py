import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from chumpy.utils import row, col
from pytorch3d import ops
from smplx import SMPLX


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

    f = 1. / ((vt01[None, :, :, 0] * vt02[None, :, :, 1] - vt01[None, :, :, 1] * vt02[None, :, :, 0]) + 1e-12)
    mask = torch.isinf(f)
    
    tangent = f[:, :, :, None] * torch.stack([
        v01[:, :, :, 0] * vt02[None, :, :, 1] - v02[:, :, :, 0] * vt01[None, :, :, 1],
        v01[:, :, :, 1] * vt02[None, :, :, 1] - v02[:, :, :, 1] * vt01[None, :, :, 1],
        v01[:, :, :, 2] * vt02[None, :, :, 1] - v02[:, :, :, 2] * vt01[None, :, :, 1]], dim=-1)

    tangent = F.normalize(tangent, dim=-1)
    normal = torch.cross(v01, v02, dim=3)
    normal = F.normalize(normal, dim=-1)

    bitangent = torch.cross(normal, tangent, dim=3) 
    bitangent = F.normalize(bitangent, dim=-1)

    # create matrix
    primrotmesh = torch.stack((tangent, bitangent, normal), dim=-1)

    return primrotmesh

def get_vertices_per_edge(num_vertices, faces):
    """
    Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()
    Adapted from https://github.com/mattloper/opendr/
    """

    vc = sp.coo_matrix(get_vert_connectivity(num_vertices, faces))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:,0] < result[:,1]] # for uniqueness
    return result

def get_vert_connectivity(num_vertices, faces):
    """
    Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12.
    Adapted from https://github.com/mattloper/opendr/
    """

    vpv = sp.csc_matrix((num_vertices,num_vertices))

    # for each column in the faces...
    for i in range(3):
        IS = faces[:,i]
        JS = faces[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T
    return vpv


if __name__ == '__main__':
    objpaths = [
        "../../../work_dirs/cache/template/dense_body.obj",
        "../../../work_dirs/cache/template/dense_hair_shoes.obj",
        "../../../work_dirs/cache/template/dense_up_and_low.obj"
    ]
    # layer 0
    # objpath = "work_dirs/cache/template/dense_body.obj"
    # layer 1
    # objpath = "work_dirs/cache/template/dense_hair_shoes.obj"
    # layer 2
    # objpath = "work_dirs/cache/template/dense_up_and_low.obj"
    for num_layer, objpath in enumerate(objpaths):
        v, vt, vi, vti = load_obj(objpath)
        v = np.array(v, dtype=np.float32)
        vt = np.array(vt, dtype=np.float32)
        vi = np.array(vi, dtype=np.int32) # np.int_ means long
        vti = np.array(vti, dtype=np.int32)
    
        vt = torch.as_tensor(vt)

        v = torch.as_tensor(v).unsqueeze(0)
        vi = torch.tensor(vi).long()
        vti = torch.tensor(vti).long()

        v0 = v[:, vi[:, 0], :].unsqueeze(2)
        v1 = v[:, vi[:, 1], :].unsqueeze(2)
        v2 = v[:, vi[:, 2], :].unsqueeze(2)

        vt0 = vt[vti[:, 0], :].unsqueeze(1)
        vt1 = vt[vti[:, 1], :].unsqueeze(1)
        vt2 = vt[vti[:, 2], :].unsqueeze(1)

        primrotmesh = compute_tbn(v0, v1, v2, vt0, vt1, vt2).view(v0.size(0), -1, 3, 3)

        final_uv = ((vt0+vt1+vt2) / 3).view(v0.size(0), -1, 2)
        final_pos = ((v0+v1+v2) / 3).view(v0.size(0), -1, 3)
        final_rot = primrotmesh
    
        np.save(f'../../../work_dirs/cache/init_faces_smplx_l{num_layer}.npy', vi.numpy())
        np.save(f'../../../work_dirs/cache/init_uv_smplx_l{num_layer}.npy', final_uv[0].numpy())
        np.save(f'../../../work_dirs/cache/init_pcd_smplx_l{num_layer}.npy', final_pos[0].numpy())
        np.save(f'../../../work_dirs/cache/init_rot_smplx_l{num_layer}.npy', final_rot[0].numpy())