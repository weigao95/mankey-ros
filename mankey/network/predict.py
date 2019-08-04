import torch
from torch.nn import functional as F


def get_integral_preds_3d_gpu(
        heatmaps,  # type: torch.Tensor,
        num_keypoints,  # type: int,
        x_dim,  # type: int,
        y_dim,  # type: int,
        z_dim,  # type: int
):  # type: (torch.Tensor, int, int, int, int) -> (torch.Tensor, torch.Tensor, torch.Tensor)
    # Reshape the tensor into 3d
    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_keypoints, z_dim, y_dim, x_dim))

    # For x_dim, it first sum the z_dim, then sum the y_dim
    # Note that heatmaps.sum() doesn't use keey_dim
    # After the sum, should be (batch_size, num_keypoints, x_dim)
    accu_x = heatmaps.sum(dim=2)
    accu_x = accu_x.sum(dim=2)
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)

    # The pointwise product
    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim), devices=[accu_x.device.index])[0].type(torch.cuda.FloatTensor)
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim), devices=[accu_y.device.index])[0].type(torch.cuda.FloatTensor)
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(z_dim), devices=[accu_z.device.index])[0].type(torch.cuda.FloatTensor)

    # Further reduce to three (batch_size, num_keypoints) tensor
    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)
    return accu_x, accu_y, accu_z


def get_integral_preds_3d_cpu(
        heatmaps,  # type: torch.Tensor,
        num_keypoints,  # type: int,
        x_dim,  # type: int,
        y_dim,  # type: int,
        z_dim,  # type: int
):  # type: (torch.Tensor, int, int, int, int) -> (torch.Tensor, torch.Tensor, torch.Tensor)
    """
    Take a normalized volumetric heatmap, get the 3d prediction from it
    The return in the range of [0-x_dim, 0-y_dim, 0-z_dim]
    :param heatmaps: The normalized heatmap, i.e., the volumetric sum should be 1
    :param num_keypoints: The input shape should be (batch_size, n_keypoints * z_dim, y_dim, x_dim)
    :param x_dim:
    :param y_dim:
    :param z_dim:
    :return: Three tensor in the shape of (batch_size, num_keypoints)
    """
    # Reshape the tensor into 3d
    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_keypoints, z_dim, y_dim, x_dim))

    # For x_dim, it first sum the z_dim, then sum the y_dim
    # Note that heatmaps.sum() doesn't use keey_dim
    # After the sum, should be (batch_size, num_keypoints, x_dim)
    accu_x = heatmaps.sum(dim=2)
    accu_x = accu_x.sum(dim=2)
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)

    # The pointwise product
    accu_x = accu_x * torch.arange(x_dim).type(torch.FloatTensor)
    accu_y = accu_y * torch.arange(y_dim).type(torch.FloatTensor)
    accu_z = accu_z * torch.arange(z_dim).type(torch.FloatTensor)

    # Further reduce to three (batch_size, num_keypoints) tensor
    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)
    return accu_x, accu_y, accu_z


def heatmap_from_predict(probability_preds, num_keypoints):  # type: (torch.Tensor, int) -> torch.Tensor:
    """
    Given the probability prediction, compute the actual probability map using softmax.
    :param probability_preds: (batch_size, n_keypoint, height, width) for 2d heatmap (batch_size, n_keypoint, height, width)
    :param num_keypoints:
    :return:
    """
    assert probability_preds.shape[1] == num_keypoints
    heatmap = probability_preds.reshape((probability_preds.shape[0], num_keypoints, -1))
    heatmap = F.softmax(heatmap, dim=2)
    heatmap = heatmap.reshape(probability_preds.shape)
    return heatmap


def heatmap2d_to_imgcoord_cpu(
        heatmap,
        num_keypoints):  # type: (torch.Tensor, int) -> (torch.Tensor, torch.Tensor)
    """
    Given the heatmap, regress the image coordinate in x (width) and y (height) direction.
    This implementation only works for 2D heatmap.
    Note that the coordinate is in [0, map_width] for x and [0, map_height] for y
    :param heatmap: (batch_size, n_keypoints, map_height, map_width) heatmap
    :param num_keypoints: For sanity check only
    :return: A tuple contains two (batch_size, n_keypoints, 1) tensor for x and y coordinate
    """
    assert heatmap.shape[1] == num_keypoints
    batch_size, _, y_dim, x_dim = heatmap.shape

    # Compute the probability on each dim
    accu_x = heatmap.sum(dim=2)
    accu_y = heatmap.sum(dim=3)

    # Element-wise product the image coord
    accu_x = accu_x * torch.arange(x_dim).type(torch.FloatTensor)
    accu_y = accu_y * torch.arange(y_dim).type(torch.FloatTensor)

    # Further reduce to three (batch_size, num_keypoints) tensor
    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    return accu_x, accu_y


def heatmap2d_to_imgcoord_gpu(
        heatmap,
        num_keypoints):  # type: (torch.Tensor, int) -> (torch.Tensor, torch.Tensor)
    assert heatmap.shape[1] == num_keypoints
    batch_size, _, y_dim, x_dim = heatmap.shape

    # Compute the probability on each dim
    accu_x = heatmap.sum(dim=2)
    accu_y = heatmap.sum(dim=3)

    # The pointwise product
    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim), devices=[accu_x.device.index])[0].type(
        torch.cuda.FloatTensor)
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim), devices=[accu_y.device.index])[0].type(
        torch.cuda.FloatTensor)

    # Further reduce to three (batch_size, num_keypoints) tensor
    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    return accu_x, accu_y


def heatmap2d_to_normalized_imgcoord_gpu(
        heatmap,
        num_keypoints):  # type: (torch.Tensor, int) -> (torch.Tensor, torch.Tensor)
    """
    Regress the normalized coordinate for x and y from the heatmap.
    The range of normalized coordinate is [-0.5, -0.5] in current implementation.
    :param heatmap:
    :param num_keypoints:
    :return:
    """
    # The un-normalized image coord
    _, _, y_dim, x_dim = heatmap.shape
    coord_x, coord_y = heatmap2d_to_imgcoord_gpu(heatmap, num_keypoints)

    # Normalize it
    coord_x *= float(1.0 / float(x_dim))
    coord_y *= float(1.0 / float(y_dim))
    coord_x -= 0.5
    coord_y -= 0.5
    return coord_x, coord_y


def depth_integration(heatmap, depth_pred):  # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Given the heatmap (normalized) and depthmap prediction, compute the
    depth value at the keypoint. There is no normalization on the depth.
    The implementation only works for 2D heatmap and depthmap.
    :param heatmap: (batch_size, n_keypoint, map_height, map_width) normalized heatmap
                    heatmap[batch_idx, keypoint_idx, :, :].sum() == 1
    :param depth_pred: (batch_size, n_keypoint, map_height, map_width) depth map
    :return: (batch_size, n_keypoint, 1) depth prediction.
    """
    assert heatmap.shape == depth_pred.shape
    n_batch, n_keypoint, _, _ = heatmap.shape
    # The pointwise product
    predict = heatmap * depth_pred

    # Aggreate on the result
    predict = predict.reshape(shape=(n_batch, n_keypoint, -1))
    return predict.sum(dim=2, keepdim=True)


def heatmap2d_to_imgcoord_argmax(heatmap):  # type: (torch.Tensor) -> (torch.Tensor, torch.Tensor)
    """
    Given the heatmap, compute the pixel with maximum heat value
    Note that this method is not differential
    :param heatmap: (batch_size, n_keypoints, map_height, map_width) heatmap
    :return:
    """
    n_batch, n_keypoint, height, width = heatmap.shape

    # Get the max value and index
    max_val, flat_idx = torch.max(heatmap.view(n_batch, n_keypoint, -1), 2)
    flat_idx_float = flat_idx.float()
    keypoint_xy_pred = torch.zeros(size=[n_batch, n_keypoint, 2], device=heatmap.get_device())
    keypoint_xy_pred[:, :, 0] = (flat_idx_float - 1) % width
    keypoint_xy_pred[:, :, 1] = torch.floor((flat_idx_float - 1) / width)
    return keypoint_xy_pred, max_val


def heatmap2d_to_normalized_imgcoord_argmax(heatmap):  # type: (torch.Tensor) -> (torch.Tensor, torch.Tensor)
    """
    Given a 2D heatmap, compute the pixel coordinate with maximum heat value
    :param heatmap: (batch_size, n_keypoints, map_height, map_width) torch.Tensor
    :return: (batch_size, n_keypoints, 2) pixel coordinate, (batch_size, n_keypoints) score
    """
    # The un-normalized image coord
    _, _, y_dim, x_dim = heatmap.shape
    coordinate, score = heatmap2d_to_imgcoord_argmax(heatmap)

    # Normalize it
    coordinate[:, :, 0] *= float(1.0 / float(x_dim))
    coordinate[:, :, 1] *= float(1.0 / float(y_dim))
    coordinate[:, :, 0] -= 0.5
    coordinate[:, :, 1] -= 0.5
    return coordinate, score
