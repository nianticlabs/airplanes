import torch


def to_minimal_representation(plane_eqn_b41: torch.Tensor) -> torch.Tensor:
    """
    Rewrite a plane equation of the form [A, B, C, D] such that Ax + By + Cz + D = 0 to be
    [-A/D, -B/D, -C/D] such that -(A/D)x - (B/D)y + -(C/D)z = 1
    """
    plane_eqn_b31 = plane_eqn_b41[:, :3] / -plane_eqn_b41[:, 3:4]
    return plane_eqn_b31


def to_explicit_representation(plane_eqn_b31: torch.Tensor) -> torch.Tensor:
    """
    Rewrite a plane equation of the form [-A/D, -B/D, -C/D] such that -(A/D)x -(B/D)y -(C/D)z = 1 to be
    [A/D, B/D, C/D, 1] such that A/Dx + B/Dy + C/Dz + 1 = 0.
    """
    plane_params_b41 = torch.cat((-plane_eqn_b31, torch.ones_like(plane_eqn_b31[:, :1])), 1)
    return plane_params_b41


def transform_plane_parameters(
    plane_params_bN1: torch.Tensor, tranformation_b44: torch.Tensor
) -> torch.Tensor:
    """
    Function to apply a 4x4 transformation to a batch of plane equations of the form Ax + By + Cz + D = 0.
    This is likely used to transform plane params from world coordinates to camera coordinates.

    Note - this also works for our planar representation of n/d, where -(Ax + By + Cx) / D = 1; in this case we
    just make our plane equation -A/Dx -B/Dy -C/Dz - 1 = 0

    Here we do the transformation in a single step, and account for potential translation (and shearing/scaling,
    although this is extremely unlikely to apply) using the following derivation:
        let p = plane equation, v = some point on the plane, M = transformation matrix, .T = transpose, ^-1 = inverse
        By definition (p.T)v = 0
        We want the new params p* such that (p*.T)Mv = 0 (the point transformed lies on the plane)
        -> (p*.T)Mv = (p.T)v -> (p*.T)M = p.T -> p*.T = (p.T)(M^-1) -> p* = (M^-1).Tp

    Basically get p* by multiplying p by the inverse transpose of M.

    We could instead convert our plane equation into a normal and point, transform them (removing translational
    component for transforming the normal), and convert back to normal and distance, but this is faster and easier.
    """

    minimal_representation = plane_params_bN1.shape[1] == 3
    if minimal_representation:
        # rewrite our minimal as explicit
        plane_params_b41 = to_explicit_representation(plane_params_bN1)
    else:
        # otherwise just use the params as is
        plane_params_b41 = plane_params_bN1

    new_transformation_b44 = torch.inverse(tranformation_b44).permute(0, 2, 1)
    transformed_plane_params_b44 = torch.matmul(new_transformation_b44, plane_params_b41)

    if minimal_representation:
        return to_minimal_representation(transformed_plane_params_b44)
    else:
        return transformed_plane_params_b44
