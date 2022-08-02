import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
import numpy as np
import json
from typing import List, Tuple
from data.labelmap import label_map
from numpy.linalg import lstsq


class Dataset(data.Dataset):
    def __init__(self, args, dataroot_path: str,
                 phase: str,
                 order: str = 'paired',
                 category: List[str] = ['dresses', 'upper_body', 'lower_body'],
                 size: Tuple[int, int] = (256, 192)):
        """
        Initialize the PyTroch Dataset Class
        :param args: argparse parameters
        :type args: argparse
        :param dataroot_path: dataset root folder
        :type dataroot_path:  string
        :param phase: phase (train | test)
        :type phase: string
        :param order: setting (paired | unpaired)
        :type order: string
        :param category: clothing category (upper_body | lower_body | dresses)
        :type category: list(str)
        :param size: image size (height, width)
        :type size: tuple(int)
        """
        super(Dataset, self).__init__()
        self.args = args
        self.dataroot = dataroot_path
        self.phase = phase
        self.category = category
        self.height = size[0]
        self.width = size[1]
        self.radius = args.radius
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

        im_names = []
        c_names = []
        dataroot_names = []

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.dataroot, c)
            if phase == 'train':
                filename = os.path.join(dataroot, f"{phase}_pairs.txt")
            else:
                filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")
            with open(filename, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        # Clothing image
        cloth = Image.open(os.path.join(dataroot, 'images', c_name))
        cloth = cloth.resize((self.width, self.height))
        cloth = self.transform(cloth)   # [-1,1]

        # Person image
        im = Image.open(os.path.join(dataroot, 'images', im_name))
        im = im.resize((self.width, self.height))
        im = self.transform(im)   # [-1,1]

        # Skeleton
        skeleton = Image.open(os.path.join(dataroot, 'skeletons', im_name.replace("_0", "_5")))
        skeleton = skeleton.resize((self.width, self.height))
        skeleton = self.transform(skeleton)

        # Label Map
        parse_name = im_name.replace('_0.jpg', '_4.png')
        im_parse = Image.open(os.path.join(dataroot, 'label_maps', parse_name))
        im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
        parse_array = np.array(im_parse)

        parse_shape = (parse_array > 0).astype(np.float32)

        parse_head = (parse_array == 1).astype(np.float32) + \
                     (parse_array == 2).astype(np.float32) + \
                     (parse_array == 3).astype(np.float32) + \
                     (parse_array == 11).astype(np.float32)

        parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                            (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["hat"]).astype(np.float32) + \
                            (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                            (parse_array == label_map["scarf"]).astype(np.float32) + \
                            (parse_array == label_map["bag"]).astype(np.float32)

        parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

        arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

        if dataroot.split('/')[-1] == 'dresses':
            label_cat = 7
            parse_cloth = (parse_array == 7).astype(np.float32)
            parse_mask = (parse_array == 7).astype(np.float32) + \
                         (parse_array == 12).astype(np.float32) + \
                         (parse_array == 13).astype(np.float32)
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        elif dataroot.split('/')[-1] == 'upper_body':
            label_cat = 4
            parse_cloth = (parse_array == 4).astype(np.float32)
            parse_mask = (parse_array == 4).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                                 (parse_array == label_map["pants"]).astype(np.float32)

            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
        elif dataroot.split('/')[-1] == 'lower_body':
            label_cat = 6
            parse_cloth = (parse_array == 6).astype(np.float32)
            parse_mask = (parse_array == 6).astype(np.float32) + \
                         (parse_array == 12).astype(np.float32) + \
                         (parse_array == 13).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                                 (parse_array == 14).astype(np.float32) + \
                                 (parse_array == 15).astype(np.float32)
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        parse_head = torch.from_numpy(parse_head)  # [0,1]
        parse_cloth = torch.from_numpy(parse_cloth)   # [0,1]
        parse_mask = torch.from_numpy(parse_mask)  # [0,1]
        parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
        parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

        # dilation
        parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
        parse_mask = parse_mask.cpu().numpy()

        # Masked cloth
        im_head = im * parse_head - (1 - parse_head)
        im_cloth = im * parse_cloth + (1 - parse_cloth)

        # Shape
        parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.width // 16, self.height // 16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.width, self.height), Image.BILINEAR)
        shape = self.transform2D(parse_shape)  # [-1,1]

        # Load pose points
        pose_name = im_name.replace('_0.jpg', '_2.json')
        with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 4))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.height, self.width)
        r = self.radius * (self.height/512.0)
        im_pose = Image.new('L', (self.width, self.height))
        pose_draw = ImageDraw.Draw(im_pose)
        neck = Image.new('L', (self.width, self.height))
        neck_draw = ImageDraw.Draw(neck)
        for i in range(point_num):
            one_map = Image.new('L', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            point_x = np.multiply(pose_data[i, 0], self.width/384.0)
            point_y = np.multiply(pose_data[i, 1], self.height/512.0)
            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                if i == 2 or i == 5:
                    neck_draw.ellipse((point_x - r*4, point_y - r*4, point_x + r*4, point_y + r*4), 'white', 'white')
            one_map = self.transform2D(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform2D(im_pose)

        im_arms = Image.new('L', (self.width, self.height))
        arms_draw = ImageDraw.Draw(im_arms)
        if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
            with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                data = json.load(f)
                shoulder_right = np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0)
                shoulder_left = np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0)
                elbow_right = np.multiply(tuple(data['keypoints'][3][:2]), self.height / 512.0)
                elbow_left = np.multiply(tuple(data['keypoints'][6][:2]), self.height / 512.0)
                wrist_right = np.multiply(tuple(data['keypoints'][4][:2]), self.height / 512.0)
                wrist_left = np.multiply(tuple(data['keypoints'][7][:2]), self.height / 512.0)
                if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                    if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                        arms_draw.line(np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')
                    else:
                        arms_draw.line(np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')
                elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                    if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                        arms_draw.line(np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')
                    else:
                        arms_draw.line(np.concatenate((elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')
                else:
                    arms_draw.line(np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(),'white', 30, 'curve')

            if self.args.height > 512:
                im_arms = cv2.dilate(np.float32(im_arms), np.ones((10, 10), np.uint16), iterations=5)
            # elif self.args.height > 256:
            #     im_arms = cv2.dilate(np.float32(im_arms), np.ones((5, 5), np.uint16), iterations=5)
            hands = np.logical_and(np.logical_not(im_arms), arms)
            parse_mask += im_arms
            parser_mask_fixed += hands

        #delete neck
        parse_head_2 = torch.clone(parse_head)
        if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
            with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                data = json.load(f)
                points = []
                points.append(np.multiply(tuple(data['keypoints'][2][:2]), self.height/512.0))
                points.append(np.multiply(tuple(data['keypoints'][5][:2]), self.height/512.0))
                x_coords, y_coords = zip(*points)
                A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                m, c = lstsq(A, y_coords, rcond=None)[0]
                for i in range(parse_array.shape[1]):
                    y = i * m + c
                    parse_head_2[int(y - 20*(self.height/512.0)):, i] = 0

        parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
        parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16), np.logical_not(np.array(parse_head_2, dtype=np.uint16))))

        if self.args.height > 512:
            parse_mask = cv2.dilate(parse_mask, np.ones((20, 20), np.uint16), iterations=5)
        # elif self.args.height > 256:
        #     parse_mask = cv2.dilate(parse_mask, np.ones((10, 10), np.uint16), iterations=5)
        else:
            parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
        parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
        im_mask = im * parse_mask_total
        parse_mask_total = parse_mask_total.numpy()
        parse_mask_total = parse_array * parse_mask_total
        parse_mask_total = torch.from_numpy(parse_mask_total)

        uv = np.load(os.path.join(dataroot, 'dense', im_name.replace('_0.jpg', '_5_uv.npz')))
        uv = uv['uv']
        uv = torch.from_numpy(uv)
        uv = transforms.functional.resize(uv, (self.height, self.width))

        labels = Image.open(os.path.join(dataroot, 'dense', im_name.replace('_0.jpg', '_5.png')))
        labels = labels.resize((self.width, self.height), Image.NEAREST)
        labels = np.array(labels)

        result = {
            'c_name': c_name,  # for visualization
            'im_name': im_name,  # for visualization or ground truth
            'cloth': cloth,  # for input
            'image': im,  # for visualization
            'im_cloth': im_cloth,  # for ground truth
            'shape': shape,  # for visualization
            'im_head': im_head,  # for visualization
            'im_pose': im_pose,  # for visualization
            'pose_map': pose_map,
            'parse_array': parse_array,
            'dense_labels': labels,
            'dense_uv': uv,
            'skeleton': skeleton,
            'm': im_mask,  # for input
            'parse_mask_total': parse_mask_total,
        }

        return result

    def __len__(self):
        return len(self.c_names)
