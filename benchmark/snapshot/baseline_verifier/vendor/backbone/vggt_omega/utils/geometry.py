# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def closed_form_inverse_se3(se3, R=None, T=None):
    """Invert one or more 3x4 or 4x4 SE(3) matrices."""
    is_numpy = isinstance(se3, np.ndarray)

    if not is_numpy and not isinstance(se3, torch.Tensor):
        raise TypeError(f"se3 must be a NumPy array or torch tensor, got {type(se3)}")
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must end in shape (4, 4) or (3, 4), got {se3.shape}")

    if R is None:
        R = se3[..., :3, :3]
    if T is None:
        T = se3[..., :3, 3:]

    if is_numpy:
        R_t = np.swapaxes(R, -1, -2)
        top_right = -np.matmul(R_t, T)
        inverted = np.broadcast_to(
            np.eye(4, dtype=se3.dtype),
            se3.shape[:-2] + (4, 4),
        ).copy()
    else:
        R_t = R.transpose(-1, -2)
        top_right = -torch.matmul(R_t, T)
        inverted = torch.eye(4, device=R.device, dtype=R.dtype).expand(
            se3.shape[:-2] + (4, 4)
        ).clone()

    inverted[..., :3, :3] = R_t
    inverted[..., :3, 3:] = top_right
    return inverted
