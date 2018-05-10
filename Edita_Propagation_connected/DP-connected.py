# -*- coding: utf-8 -*-
# Copyright (C) 2016 Yuki Endo

import argparse
import json
import time
import numpy.random
import sklearn.feature_extraction
import skimage.segmentation
import cv2
import math
import DNN9
import pydensecrf.densecrf as dcrf

class DeepProp(object):

    def __init__(self, img_path):
        self.img = cv2.imread(img_path)

        self.patch_radius = 4
        self.t_data_ratio = 0.1
        self.sp_ratio = 1

        self.lmbd = 10.
        self.affinity_weight = 0.01

        self.color2label = dict()
        self.label2color = dict()

    def recoloring(self, strk_path, out_path="./out.jpg"):
        # load images
        
        img = self.img
        strk = cv2.imread(strk_path, -1)
        print("load input")

        # extract patch features
        reflected_img = cv2.copyMakeBorder(
            img, self.patch_radius, self.patch_radius, self.patch_radius, self.patch_radius, cv2.BORDER_REFLECT_101)
        patch_features = sklearn.feature_extraction.image.extract_patches_2d(
            reflected_img / 255., (self.patch_radius * 2 + 1, self.patch_radius * 2 + 1))

        h, w = img.shape[:2]
        strk_point = self.find_strk_point(strk)
        self.color2label = {k: v for v, k in enumerate(strk_point.keys())}
        self.label2color = {k: v for k, v in enumerate(strk_point.keys())}
        label_dim = len(self.color2label.keys())

        # prepare training data
        X_patch = list()
        Y_label = list()
        maxdis = 255.

        t0 = time.time()
        time1 = time2 = 0.
        for y, row in enumerate(strk[:, :, 3]):
            for x, val in enumerate(row):
                if val == 0 or numpy.random.rand() > self.t_data_ratio:
                    continue
                color_str = json.dumps(strk[y, x][:3].tolist())

                # 取样本点的坐标
                t1 = time.time()
                min_interval = (1 / 20.) * math.sqrt(h * h + w * w)
                coord_simple = dict()
                for one_label in strk_point.keys():
                    coord_simple[one_label] = list()
                    for one_strk_coord in strk_point[one_label]:
                        a = numpy.random.randint(len(one_strk_coord))
                        coord_simple[one_label].append(one_strk_coord[a])
                        for point_coord in one_strk_coord:
                            if numpy.random.rand() < 0.01:
                                flag = True
                                for coord in coord_simple[one_label]:
                                    if numpy.sqrt((coord[0] - point_coord[0]) ** 2 + (
                                            coord[1] - point_coord[1]) ** 2) < min_interval:
                                        flag = False
                                        break
                                if flag:
                                    coord_simple[one_label].append(point_coord)
                time1 += time.time() - t1                

                # 求距离图
                t2 = time.time()
                disMap = numpy.zeros((9,9,label_dim), numpy.float32)
                for i, y_c in enumerate(range(y-4,y+5)):
                    for j, x_c in enumerate(range(x-4,x+5)):
                        for k in range(label_dim):
                            mindis = 1.
                            for ii in coord_simple[self.label2color[k]]:
                                dis = math.sqrt((y_c-ii[0])**2+(x_c-ii[1])**2)/maxdis
                                dis = dis if dis<=1 else 1
                                if dis < mindis:
                                    mindis = dis
                            disMap[i][j][k]=mindis

                X_patch.append(numpy.append(patch_features[y * w + x], disMap, axis=2))
                Y_label.append(self.color2label[color_str])
                time2 += time.time() - t2

        Y_label1 = numpy.zeros((len(Y_label), label_dim))
        for i in range(len(Y_label)):
            Y_label1[i][Y_label[i]] = 1
        
        print("prepare training data Time:", time1)
        print("prepare training data Time:", time2)
        print("Time_preparing training data:", time.time() - t0)
        print("Training size:", len(Y_label))

        # training
        t0 = time.time()
        dnn = DNN9.DNN(x_dim=9*9*(3+label_dim), y_dim=label_dim, patch_radius=4, channel=3+label_dim)
        dnn.train(X_patch,Y_label1)
        print("Time_training:", time.time() - t0)

        # superpixel segmentationd
        segments = skimage.segmentation.slic(img, n_segments=int(
            w * h * self.sp_ratio), compactness=50., enforce_connectivity=True)

        # create test data from superpixels
        spid2center = dict()
        spid2pixnum = dict()
        for y in range(h):
            for x in range(w):
                spid2center[segments[y, x]] = spid2center.get(
                    segments[y, x], numpy.zeros(2, numpy.float)) + numpy.array([y, x], numpy.float)
                spid2pixnum[segments[y, x]] = spid2pixnum.get(
                    segments[y, x], 0) + 1

                
        # 
        t0 = time.time()
        # 取样本点的坐标
        min_interval = (1 / 50.) * math.sqrt(h * h + w * w)
        coord_simple = dict()
        for one_label in strk_point.keys():
            coord_simple[one_label] = list()
            for one_strk_coord in strk_point[one_label]:
                a = numpy.random.randint(len(one_strk_coord))
                coord_simple[one_label].append(one_strk_coord[a])                
                for point_coord in one_strk_coord:
                    if numpy.random.rand() < 0.01:
                        flag = True
                        for coord in coord_simple[one_label]:
                            if numpy.sqrt((coord[0] - point_coord[0]) ** 2 + (
                                    coord[1] - point_coord[1]) ** 2) < min_interval:
                                flag = False
                                break
                        if flag:
                            coord_simple[one_label].append(point_coord)

        # 求距离图
        disMap = numpy.zeros((h, w, label_dim), numpy.float32)
        for i in range(h):
            for j in range(w):
                for k in range(label_dim):
                    mindis = 1.
                    for ii in coord_simple[self.label2color[k]]:
                        dis = math.sqrt((i - ii[0]) ** 2 + (j - ii[1]) ** 2) / maxdis
                        dis = dis if dis <= 1 else 1
                        if dis < mindis:
                            mindis = dis
                    disMap[i][j][k] = mindis
      
        reflected_disMap = cv2.copyMakeBorder(
            disMap, self.patch_radius, self.patch_radius, self.patch_radius, self.patch_radius, cv2.BORDER_REFLECT_101)
        disMap_features = sklearn.feature_extraction.image.extract_patches_2d(
            reflected_disMap, (self.patch_radius * 2 + 1, self.patch_radius * 2 + 1))

        X_patch = list()
        spid2featureid = dict()
        for spid in spid2center.keys():
            spid2center[spid] = spid2center[spid] / float(spid2pixnum[spid])
            y = int(round(spid2center[spid][0]))
            x = int(round(spid2center[spid][1]))
            X_patch.append(numpy.append(patch_features[y * w + x], disMap_features[y * w + x], axis=2))
            spid2featureid[spid] = len(X_patch) - 1
        print("Time_preparing estimation:", time.time() - t0)

        # estimation
        t0 = time.time()
        Y_label = dnn.estimate(X_patch)
        print("Time_estimation:", time.time() - t0)

        # segmentation(refinement)        
        label_dim = len(self.color2label.keys())
        p_img = numpy.zeros((label_dim, h, w, 3), numpy.uint8)
        label_pro = numpy.zeros((label_dim, h, w), numpy.float32)
        for y in range(h):
            for x in range(w):
                spid = segments[y, x]
                featureid = spid2featureid[spid]
                probs = Y_label[featureid]
                for i in range(label_dim):
                    pColor = [probs[i] * 255, probs[i] * 255, probs[i] * 255]
                    p_img[i, y, x] = numpy.uint8(pColor)
                    label_pro[i, y, x] = -numpy.log(probs[i] + 0.00001)
                if strk[y][x][3] != 0:
                    color_str = json.dumps(strk[y, x][:3].tolist())
                    dim = self.color2label[color_str]
                    for i in range(label_dim):
                        label_pro[i, y, x] = 99999.
                    label_pro[dim, y, x] = 0.

        t0 = time.time()
        d = dcrf.DenseCRF2D(w, h, label_dim)
        U = label_pro.reshape((label_dim, -1))
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=2)
        d.addPairwiseBilateral(sxy=60, srgb=13, rgbim=img, compat=7)
        Q = d.inference(6)
        map = numpy.argmax(Q, axis=0).reshape((h, w))

        crf_mask = numpy.zeros((label_dim, h, w, 3), numpy.uint8)
        crf_img = numpy.zeros((label_dim, h, w, 3), numpy.uint8)
        for y in range(h):
            for x in range(w):
                for i in range(label_dim):
                    if map[y, x] == i:
                        mColor = [255, 255, 255]
                        iColor = img[y, x]
                    else:
                        mColor = [0, 0, 0]
                        iColor = [0, 0, 0]
                    crf_mask[i, y, x] = numpy.uint8(mColor)
                    crf_img[i, y, x] = numpy.uint8(iColor)
        print("Time_crf refinement:", time.time() - t0)

        # save segmentation results        
        for i in range(label_dim):
            cv2.imwrite('data/out' + str(i) + '.png', p_img[i])
            cv2.imwrite('data/out' + str(i) + 'crf_mask.png', crf_mask[i])
            cv2.imwrite('data/out' + str(i) + 'crf_img.png', crf_img[i])

        
        # coloring
        c_img = numpy.zeros((h, w, 3), numpy.uint8)
        for y in range(h):
            for x in range(w):
                label = map[y, x]
                color = numpy.array(json.loads(
                    self.label2color[label]), numpy.float)
                if color.dot(color) == 0.:
                    color = img[y, x]
                res_color = color
                c_img[y, x] = numpy.uint8(res_color)

        c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2Lab)
        img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        res_img = numpy.c_[img_Lab[:, :, :1], c_img[:, :, 1].reshape(h, w, 1), c_img[
                                                                               :, :, 2].reshape(h, w, 1)]
        res_img = cv2.cvtColor(res_img, cv2.COLOR_Lab2BGR)

        #save recoloring results
        cv2.imwrite(out_path, res_img)


    def find_strk_point(self, strk):
        h, w = strk.shape[:2]
        visited = numpy.zeros((h, w), numpy.uint8)
        queue = list()
        strk_point = dict()
        for y in range(h):
            for x in range(w):
                if strk[y][x][3] == 0:
                    visited[y][x] = 1
                    continue
                if visited[y][x] == 0:
                    color_str = json.dumps(strk[y, x][:3].tolist())
                    strk_point[color_str] = strk_point.get(
                        color_str, list())
                    s_point = list()
                    queue.append([y, x])
                    while len(queue) != 0:
                        point_current = queue.pop()
                        if strk[point_current[0]][point_current[1]][3] != 0 and visited[point_current[0]][point_current[1]] == 0:
                            s_point.append(point_current)
                            visited[point_current[0]][point_current[1]] = 1
                            for p_y in range(point_current[0]-1,point_current[0]+2):
                                for p_x in range(point_current[1] - 1, point_current[1] + 2):
                                    if p_y >= 0 and p_x >= 0 and p_y < h and p_x < w:
                                        if strk[p_y][p_x][3] != 0 and visited[p_y][p_x] == 0:
                                            queue.append([p_y, p_x])
                    strk_point[color_str].append(s_point)
        return strk_point


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', help='file path of input image',
                        default='data/3.jpg')
    parser.add_argument('-s', help='file path of user stroke',
                        default='data/3.png')
    parser.add_argument(
        '-o', help='file path of output image', default='data/out.png')
    args = parser.parse_args()

    DP = DeepProp(img_path=args.i)
    t0 = time.time()
    DP.recoloring(strk_path=args.s, out_path=args.o)
    print("Total Time:", time.time() - t0)

if __name__ == '__main__':
    main()
