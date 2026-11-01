import math
import os.path as osp
import json

import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
import torch


# from reid import datasets, models
# from reid.utils.data import transforms as T
# from reid.utils.data.preprocessor import Preprocessor
# from reid.utils.osutils import set_paths
# from reid.utils.serialization import load_checkpoint
# from reid.models.CompactBilinearPooling_dsybaik import CompactBilinearPooling

# Source: https://github.com/yuminsuh/part_bilinear_reid/blob/master/vis_featmap.ipynb
from torchreid.utils import Writer, Logger
from torchreid.utils.constants import PARTS


def flatten(maps):
    flattened = np.transpose(maps, (1, 0, 2, 3)).reshape(maps.shape[1], int(maps.size / maps.shape[1]))
    return flattened


def organize(flattened, num_map, feat_dim, h, w):
    maps = flattened.reshape(feat_dim, num_map, h, w)
    maps = np.transpose(maps, (1, 0, 2, 3))
    return maps


def feat_to_color(maps):
    num_img, dim_feat, h, w = maps.shape
    maps_flatten = flatten(maps)
    maps_flatten_reduced = PCA(n_components=3).fit_transform(maps_flatten.transpose())
    maps_reduced = organize(maps_flatten_reduced.transpose(), num_img, 3, h, w)
    return maps_reduced


def normalize_01(m):
    max_value = m.max()
    min_value = m.min()
    r = max_value - min_value
    n = (m - min_value) / r
    return n


def mapwise_normalize(map1, feats):
    num_sample, num_channel, h, w = map1.shape # FIXME TOO MANY VALUES TO UNPACK WITH INDEITITy MASK
    # mode 1
    # norms = np.linalg.norm(np.sqrt(np.sum(map1**2, axis=(2,3))), axis=1)
    # normalized = map1
    # mode 2
    norms = np.sqrt(np.linalg.norm(feats, axis=1))
    normalized = map1 / np.reshape(norms, (num_sample, 1, 1, 1))
    return normalized


# TODO output big grid of n identities horizontally and m samples per identity vertically
# TODO test without norm stuff
# TODO refactor into beautiful code
# TODO cannot apply PCA batch by batch, should be global
def visualize_pca_multi(maps_all, feats, pids, tag):
    maps_all_reduced = []
    for maps in maps_all[1:]:
        if len(maps.shape) == 4:
            maps_all_reduced.append(feat_to_color(mapwise_normalize(maps, feats)))
    num_person = maps_all_reduced[0].shape[0]

    num_samples = len(pids)
    num_samples_per_id = 0
    for i, pid in enumerate(pids):
        if pid != pids[0]:
            num_samples_per_id = i
            break



    n_rows = num_samples_per_id
    n_cols = math.ceil(num_samples / num_samples_per_id)
    fig = plt.figure(figsize=(n_cols*4, n_rows*2), constrained_layout=False)
    outer_grid = fig.add_gridspec(n_rows, n_cols)

    # count = 0
    # for row in range(4):
    #     for col in range(5):
    #         print("grid {}-{}".format(row, col))
    #         # gridspec inside gridspec
    #         inner_grid = outer_grid[row, col].subgridspec(1, 3)
    #         axs = inner_grid.subplots()  # Create all subplots for the inner grid.
    #         triplet = triplets[count]
    #         pos, anc, neg, pos_dist, neg_dist = triplet
    #         ax1, ax2, ax3 = axs[0], axs[1], axs[2]
    #         show_instance(ax1, pos, pos_dist, green)
    #         show_instance(ax2, anc, 0, black)
    #         show_instance(ax3, neg, neg_dist, red)
    #         count += 1

    # for person_idx in range(min(num_person, num_vis)):
    person_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if person_idx < num_person:
                # print(person_idx)
                # fig, ax = plt.subplots(1, len(maps_all))
                #         fig.set_size_inches((10,10))
                inner_grid = outer_grid[row, col].subgridspec(1, 3)
                axs = inner_grid.subplots()  # Create all subplots for the inner grid.
                for idx, maps_reduced in enumerate([maps_all[0]] + maps_all_reduced):
                    axs[idx].imshow(normalize_01(np.transpose(maps_reduced[person_idx, ::-1, :, :], (1, 2, 0))))
                    axs[idx].set_axis_off()
                person_idx += 1
    # plt.show()

    # plt.tight_layout()
    plt.tight_layout(pad=1.30, h_pad=1.6, w_pad=1.6)
    Logger.current_logger().add_figure("features_part_maps_{}".format(tag), fig, False)
    plt.close(fig)

#         fig.savefig(savepath.format(save_id, person_idx))


def display_feature_maps(embeddings_dict, spatial_features, body_part_masks, imgs_path, pids):
    # TODO call at test time: display top 10 ranking for 5 queries? Take 10 random pids with 1 samples in query and
    #  find 9 corresponding samples with same pid in gallery

    # TODO put config
    # TODO fix with new model output

    writer = Writer.current_writer()
    def extract_images(imgs_path):
        imgs = []
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 256)) # TODO size = config
            img = img / 255
            imgs.append(img)

        imgs = np.asarray(imgs)
        imgs = np.transpose(imgs, axes=[0, 3, 1, 2])
        return imgs

    if writer.engine_state.epoch == writer.engine_state.max_epoch-1 and writer.engine_state.batch < 10 and writer.cfg.test.vis_feature_maps: # TODO move away
        # TODO
        body_parts_features = embeddings_dict[PARTS]
        body_part_masks = torch.unsqueeze(body_part_masks, 2)  # [N, M, 1, Hf, Wf]
        spatial_features = torch.unsqueeze(spatial_features, 1)  # [N, 1, D, Hf, Wf]
        images_np = extract_images(imgs_path)
        if len(body_parts_features.shape) == 3:
            body_parts_features = body_parts_features.flatten(1, 2)
        body_parts_features_np = body_parts_features.squeeze().detach().cpu().numpy().copy()
        body_part_masks_np = body_part_masks.squeeze().detach().cpu().numpy().copy()
        spatial_features_np = spatial_features.squeeze().detach().cpu().numpy().copy()
        visualize_pca_multi([images_np, spatial_features_np, body_part_masks_np], body_parts_features_np, pids, "e_{}_b_{}".format(writer.engine_state.epoch, writer.engine_state.batch))
