import os
import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import SimpleITK as sitk
import shutil
import json
from scipy.ndimage import gaussian_filter
from time import time
from sklearn.impute import KNNImputer
import pandas as pd



def proj_calc(p1, p2, p3):
    l2 = np.sum((p1 - p2) ** 2)
    param = np.sum((p3 - p1) * (p2 - p1)) / (l2 + 1e-8)
    #         if param > 1 or param < 0:
    #             print('p3 does not project onto p1-p2 line segment')

    param = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / (l2 + 1e-8)))
    p_proj = p1 + param * (p2 - p1)

    l_proj = np.sqrt(np.sum((p_proj - p1) ** 2))
    h = np.sqrt(np.sum((p3 - p_proj) ** 2))
    return l_proj, h


def proj_coord(i, proj, ctr_neighbor_dist):
    if i == 0:
        return proj
    return ctr_neighbor_dist[i - 1] + proj


def ses_smoothing(st_mat, alpha=0.1, ligher_edge=False):
    st_mat_2 = np.zeros(st_mat.shape, dtype=st_mat.dtype)
    st_mat_2[-1, :] = st_mat[-1, :].copy()

    for s_i in np.arange(st_mat.shape[0] - 1)[::-1]:
        st_mat_2[s_i, :] = (alpha * st_mat_2[s_i + 1, :] + (1 - alpha) * st_mat[s_i, :]).copy()

    #     for s_i in range(1, st_mat_2.shape[0]):
    #         st_mat_2[s_i,:] = (alpha * st_mat_2[s_i-1,:] + (1 - alpha) * st_mat_2[s_i,:]).copy()

    if ligher_edge:
        st_mat_2[:10, :] = st_mat_2[10:20, :]

    return st_mat_2


def break_rib(approx, key, k=3):
    if 'R' in key:
        cut_idx = np.argmin(approx[:, 0])
    elif 'L' in key:
        cut_idx = np.argmax(approx[:, 0])
    else:
        raise Exception("key not provide right, has to be L# or R#")

    dist_lst = []
    arr = np.concatenate([approx[:cut_idx - k], approx[cut_idx + k:]], axis=0)

    for i in range(arr.shape[0]):
        p1 = approx[cut_idx]
        p2 = arr[i]
        dist_lst.append(np.sum((p1 - p2) ** 2))
    dist_lst = np.array(dist_lst)
    opposite_idx = np.argmin(dist_lst)

    for i in range(approx.shape[0]):
        if (approx[i] == arr[opposite_idx]).sum() == 2:
            break
    idx_ls = np.array([cut_idx, i])
    l_idx = np.argmax(idx_ls)
    ctr1 = np.concatenate([approx[:idx_ls[l_idx - 1] + 1], approx[idx_ls[l_idx]:]], axis=0)
    ctr2 = approx[idx_ls[l_idx - 1]: idx_ls[l_idx] + 1]
    return ctr1, ctr2


def expand_ctr(approx):
    if approx.shape[0] % 2 == 1:
        lp_range = approx.shape[0] - 1
    else:
        lp_range = approx.shape[0] - 1

    ctr_lst = [approx[0]]
    for i in range(0, lp_range):
        p1 = approx[i]
        p3 = approx[i + 1]
        p2 = (p1 + p3) / 2;
        p2 = p2.astype(np.int32)
        #     ctr_lst.append(p1)
        ctr_lst.append(p2)
        ctr_lst.append(p3)
    ctr_arr = np.array(ctr_lst)
    return ctr_arr


def exclude_outliers(mat):
    e1 = int(mat.shape[0] / 3)
    e2 = int(mat.shape[0] / 2)
    val = mat[e2, :].sum()
    imax = 0
    mat_2 = mat.copy()
    for i in range(e1):
        if mat[i].sum() / val > 3:
            imax = i
    if imax == 0:
        mat_2[0, :] = mat[1, :].copy()
    else:
        mat_2[:imax + 1, :] = mat[imax + 1:2 * (imax + 1), :].copy()

    return mat_2


def smooth_ctrLine(mat, th=0.95, k=3):
    mat_2 = mat.copy()
    for i in range(1, mat.shape[0]):
        msk = mat_2[i] / (mat_2[i - 1] + 1e-8) <= th
        if msk.sum() != 0:
            mat_2[i][msk] = (mat_2[i - k:i].sum(axis=0) / k)[msk]
    return mat_2


def KNN_smooth_border(mat, mask, k=5):
    imputer = KNNImputer(n_neighbors=k, weights='distance')
    img = mat.copy().astype(np.float64)
    img[mask == 0] = np.nan
    img = imputer.fit_transform(img)
    img = img.astype(mat.dtype)
    return img



## image & annotation loading
ann_path = \
    '/home/flute/ShengNAS2/SharedProjectData/MICCAI_2022/VinDr_RibCXR_Dataset/Annotations/val/Vindr_RibCXR_val_mask.json'
img_dir = \
    '/home/flute/ShengNAS2/SharedProjectData/MICCAI_2022/VinDr_RibCXR_Dataset/data/val/img'
save_dir = \
    '/home/flute/ShengNAS2/SharedProjectData/MICCAI_2022/VinDr_RibCXR_Dataset/data/val/img_rib_removed'
if not os.path.exists(save_dir): os.makedirs(save_dir)

break_lst = ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8',
             'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', ]

key_lst = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10',
           'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10']


with open(ann_path, 'r') as f:
    ann = json.load(f)

rib_hist = []
# for img_num in range(196):
# for img_num in np.arange(192)[::-1]:
for img_num in [48]:
    t_start = time()
    print('currently processing %d img.' % img_num)
    img_path = os.path.join(img_dir, 'VinDr_RibCXR_val_%s.png' % str(img_num).rjust(3, '0'))
    save_path = os.path.join(save_dir, 'VinDr_RibCXR_val_%s.png' % str(img_num).rjust(3, '0'))

    img = cv2.imread(img_path)
    img_2 = img.copy()
    img_3 = img.copy()

    k_ext = 5
    k_org = 3

    # knn_s = 5
    knn_k = 3

    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_img_2 = g_img.copy()

    deri_st_mat_hist = {}
    # take only img of #0 as a starter trial
    for r_num, key in enumerate(key_lst):
        print('\ncurrently processing %s ribs, %d left to go...' % (key, len(key_lst) - r_num - 1))

        coords = ann[key][str(img_num)]
        ctr = []
        for coord in coords:
            x, y = coord['x'], coord['y']
            ctr.append([x, y])
        ctr = np.array(ctr, dtype=np.int32)

        ## img preprocessing
        x_min, x_max, y_min, y_max = 10000, 0, 10000, 0

        for i in range(len(ctr)):
            x, y = ctr[i]
            if x < x_min: x_min = x
            if x > x_max: x_max = x

            if y < y_min: y_min = y
            if y > y_max: y_max = y

        ## bbox in the form of xywh
        margin = 5
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        wide_bbox = [x_min - margin, y_min - margin,
                     x_max - x_min + 2 * margin, y_max - y_min + 2 * margin]

        # find the complete rib ctr
        bg_img_2 = cv2.cvtColor(g_img_2, cv2.COLOR_GRAY2BGR)
        gray_patch_complete = img_2[wide_bbox[1]:wide_bbox[1] + wide_bbox[3],
                              wide_bbox[0]:wide_bbox[0] + wide_bbox[2]]

        t_ext = bg_img_2.copy()
        cv2.fillPoly(t_ext, pts=[ctr], color=(255, 0, 255))
        cv2.drawContours(t_ext, [ctr], -1, (255, 0, 0), k_ext)
        rib_ext = bg_img_2.copy()
        rib_ext[t_ext == bg_img_2] = 0
        patch_ext = rib_ext[wide_bbox[1]:wide_bbox[1] + wide_bbox[3],
                    wide_bbox[0]:wide_bbox[0] + wide_bbox[2], :]
        gray_patch_ext = cv2.cvtColor(patch_ext, cv2.COLOR_BGR2GRAY)

        ## find sparse ctr of L5 rib
        ret, thresh = cv2.threshold(gray_patch_ext, 1, 255, 0)
        contours, hierarchy = \
            cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        # calc arclentgh
        arclen = cv2.arcLength(cnt, True)
        # do approx
        eps = 0.0005
        epsilon = arclen * eps
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx = expand_ctr(np.squeeze(approx, axis=1))

        if key in break_lst:
            ctr1, ctr2 = break_rib(approx, key)
            ctrs = [ctr1, ctr2]
            if ctr1.shape[0] > ctr2.shape[0]:
                scls = [4, 10]
            else:
                scls = [10, 4]
        else:
            ctrs = [approx]
            scls = [4]

        for ctr_ct, (segment_ctr, scl) in enumerate(zip(ctrs, scls)):
            #             print('  %s has %d segments to process, currently processing the %d one;'%(key, len(ctrs), ctr_ct+1))
            #             print('  contour is of %d points;'%len(segment_ctr))

            t_org = gray_patch_complete.copy()
            cv2.fillPoly(t_org, pts=[segment_ctr], color=(255, 0, 255))
            cv2.drawContours(t_ext, [segment_ctr], -1, (255, 0, 0), k_org)
            rib_org = gray_patch_complete.copy()
            rib_org[t_org == gray_patch_complete] = 0
            gray_patch_org = cv2.cvtColor(rib_org, cv2.COLOR_BGR2GRAY)

            t_ext = gray_patch_complete.copy()
            cv2.fillPoly(t_ext, pts=[segment_ctr], color=(255, 0, 255))
            cv2.drawContours(t_ext, [segment_ctr], -1, (255, 0, 0), k_ext)
            rib_ext = gray_patch_complete.copy()
            rib_ext[t_ext == gray_patch_complete] = 0
            gray_patch_ext = cv2.cvtColor(rib_ext, cv2.COLOR_BGR2GRAY)

            knn_mask_patch = np.ones(gray_patch_ext.shape, dtype=np.int32)


            ## process rib segment coordinates in XY System
            y_idxes, x_idxes = np.ma.where(gray_patch_ext != 0)  ### attention there is a flip here!!!!!
            segment_xy_ls = []
            for x, y in zip(list(x_idxes), list(y_idxes)):
                segment_xy_ls.append((x, y))
            segment_xy = np.array(segment_xy_ls, dtype=np.int64)

            ## arrange ctr Tangent line set and calculate culmulate distance
            line_sets = []
            ctr_neighbor_dist = []
            d = 0
            for i in range(segment_ctr.shape[0]):
                if i == segment_ctr.shape[0] - 1:
                    p1 = segment_ctr[i]
                    p2 = segment_ctr[0]
                else:
                    p1 = segment_ctr[i]
                    p2 = segment_ctr[i + 1]

                line_sets.append([p1, p2])
                d += np.sqrt(np.sum((p1 - p2) ** 2))
                ctr_neighbor_dist.append(d.copy())

            ## ST SPACE TREANSFORMATION
            #             print('  XY2ST SPACE Transformation started:')
            segment_st_ls = []
            tmax = int(ctr_neighbor_dist[-1]) + 1
            smax = 0
            st_mat = np.zeros((2000, tmax), dtype=img.dtype)

            for ct, p3 in enumerate(segment_xy):
                #                 if ct%10000==0: print('    %d out of %d is processed;'%(ct, segment_xy.shape[0]))

                d_min = 10000
                idx = 0
                l_proj = 0
                for i in range(len(line_sets)):
                    p1 = line_sets[i][0]
                    p2 = line_sets[i][1]

                    l_proj_c, d = proj_calc(p1, p2, p3)

                    if d < d_min:
                        d_min = np.copy(d)
                        l_proj = np.copy(l_proj_c)
                        idx = np.copy(i)

                t = int(proj_coord(idx, l_proj, ctr_neighbor_dist))
                s = int(d_min)
                if s > smax: smax = np.copy(s)
                segment_st_ls.append((s, t))  ### record to match the segment_xy set they are 121 correspondence
                if s <= k_ext: knn_mask_patch[y, x] = 0

                x, y = p3
                st_mat[s, t] = gray_patch_ext[y, x].copy()
            st_mat = st_mat[:smax + 1, :].astype(np.int64)
            #             print('    Transforming done! st_mat of shape %s'%str(st_mat.shape))

            ## DI/Ds calculation
            deri_st_mat = np.zeros(st_mat.shape, dtype=np.int64)
            deri_st_mat[0, :] = st_mat[0, :].copy()

            for i in range(1, st_mat.shape[0]):
                deri_st_mat[i, :] = (st_mat[i, :] - st_mat[i - 1, :]).copy()

            #   smooth DI/Ds
            if key in ['R1', 'L1']: scl = 10
            if key in ['R10', 'L9', 'L10']: scl = 20

            smoothed_deri_st_mat = gaussian_filter(deri_st_mat.astype(np.float64),
                                                   sigma=(0, st_mat.shape[-1] * scl), mode='nearest')
            smoothed_st_mat = np.zeros(st_mat.shape, dtype=np.float64)
            smoothed_st_mat[0, :] = smoothed_deri_st_mat[0, :].copy()

            ## reintegrate smooth DI/Ds
            for i in range(1, st_mat.shape[0]):
                smoothed_st_mat[i, :] = (smoothed_deri_st_mat[i, :] + smoothed_st_mat[i - 1, :]).copy()

            smoothed_st_mat = exclude_outliers(smoothed_st_mat)
            smoothed_st_mat = smooth_ctrLine(smoothed_st_mat, .95, 5)

            ## ST -> XY space transformation
            gray_patch_ext_bone = np.zeros(gray_patch_ext.shape, dtype=smoothed_st_mat.dtype)
            for (s, t), (x, y) in zip(segment_st_ls, segment_xy):
                smoothed_I = smoothed_st_mat[s, t].copy()
                gray_patch_ext_bone[y, x] = np.copy(smoothed_I)

            gray_patch_ext_bone[gray_patch_ext_bone < 0] = 0
            gray_patch_ext_bone = gray_patch_ext_bone.astype('uint8')
            gray_patch_org_bone = gray_patch_ext_bone.copy()
            gray_patch_org_bone[gray_patch_org == 0] = 0
            # knn_mask_patch[gray_patch_org == 0] = 1

            rib = np.zeros(g_img_2.shape, dtype=g_img_2.dtype)
            rib[wide_bbox[1]:wide_bbox[1] + wide_bbox[3], wide_bbox[0]:wide_bbox[0] + wide_bbox[2]] = \
                gray_patch_org_bone

            knn_mask_img = np.ones(g_img_2.shape, dtype=np.int32)
            knn_mask_img[wide_bbox[1]:wide_bbox[1] + wide_bbox[3], wide_bbox[0]:wide_bbox[0] + wide_bbox[2]] = \
                knn_mask_patch

            rib_hist.append(rib)

            g_img_2 = (g_img_2 - rib).copy()
            g_img_2 = KNN_smooth_border(g_img_2, knn_mask_img, knn_k)
            g_img_2[g_img_2 < 0] = 0

        #         fig = plt.figure(figsize=(20,20))
        #         ax = fig.add_subplot(111)
        #         ax.imshow(g_img_2, cmap='gray')
        #         plt.show()

        cv2.imwrite(save_path, g_img_2)
    print('derib took %.1f mins ...\n' % ((time() - t_start) / 60))