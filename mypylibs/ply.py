"""Minimal binary PLY writer (no Open3D in the course image)."""

import numpy as np


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a coloured point cloud. xyz (n, 3) float, rgb (n, 3) uint8.

    OpenCV cameras look down +z with y pointing down; we flip y and z so viewers
    (MeshLab, CloudCompare) show the scene upright.
    """
    vertex = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex['x'], vertex['y'], vertex['z'] = xyz[:, 0], -xyz[:, 1], -xyz[:, 2]
    vertex['red'], vertex['green'], vertex['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(xyz)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'end_header\n')
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertex.tobytes())
