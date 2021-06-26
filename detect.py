from __future__ import division
from __future__ import print_function

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import matplotlib.cm as cm
import matplotlib.colors as mcolors



import numpy as np
import cv2

#######################################################################

# Projection matrix from rect camera coord to image2 coord
#self.P = calibs['P2']
P= np.array([[649.65825827, 0, 302.21275333, 0.],
             [0, 656.18334152, 244.27286533, 0.],
             [0, 0, 1., 0.]])
#self.P = np.reshape(self.P, [3,4])

# Rigid transform from Velodyne coord to reference camera coord
        
V2C = np.array([[0.08088629, -0.99590131, -0.04047205, -0.15610122],
             [0.06293044, 0.04562682,-0.9969744, -0.3785559 ],
             [0.99473472, 0.07809463, 0.06636309, -0.59070911]])
                
#V2C = np.reshape(V2C, [3,4])

# Rotation from reference camera coord to rect camera coord
R0 = np.array([[1, 0, 0],
               [0, 1, 0 ],
               [0, 0, 1]])  
#self.R0 = np.reshape(self.R0,[3,3])

# Camera intrinsics and extrinsics
c_u = P[0,2]
c_v = P[1,2]
f_u = P[0,0]
f_v = P[1,1]

def cart2hom(pts_3d): 
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
    return pts_3d_hom

# ===========================
# ------- 3d to 3d ----------
# ===========================
def project_velo_to_ref(pts_3d_velo):  
    pts_3d_velo = cart2hom(pts_3d_velo) # nx4
    return np.dot(pts_3d_velo, np.transpose(V2C))

def project_ref_to_rect(pts_3d_ref):  
    ''' Input and Output are nx3 points '''
    return np.transpose(np.dot(R0, np.transpose(pts_3d_ref)))

def project_velo_to_rect( pts_3d_velo): 
    pts_3d_ref = project_velo_to_ref(pts_3d_velo)
    return project_ref_to_rect(pts_3d_ref)

# ===========================
# ------- 3d to 2d ----------
# ===========================
def project_rect_to_image(pts_3d_rect): 
    ''' Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = np.dot(pts_3d_rect, np.transpose(P)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

def project_velo_to_image(pts_3d_velo): 
    ''' Input: nx3 points in velodyne coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = project_velo_to_rect(pts_3d_velo)
    return project_rect_to_image(pts_3d_rect)

def generate_dispariy_from_velo(pc_velo, height, width):
    pts_2d = project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_velo[:, 0] > 2)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width)) - 1
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    max_depth=imgfov_pc_rect[:,2].max()
    for i in range(imgfov_pts_2d.shape[0]):
        #depth = imgfov_pc_rect[i, 2]
        #print(max_depth, imgfov_pc_rect[i, 2] )
        depth =max_depth-imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    return depth_map
##################################################################################

kitti_weights = '/content/PyTorch-YOLOv3-kitti/weights/yolov3-kitti.weights'

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='/content/PyTorch-YOLOv3-kitti/data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='/content/PyTorch-YOLOv3-kitti/config/yolov3-kitti.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default=kitti_weights, help='path to weights file')
parser.add_argument('--class_path', type=str, default='/content/PyTorch-YOLOv3-kitti/data/kitti.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print('Config:')
print(opt)





cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)
print('model path: ' +opt.weights_path)
if cuda:
    model.cuda()
    print("using cuda model")

model.eval() # Set in evaluation mode

dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path) # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print('data size : %d' % len(dataloader) )
print ('\nPerforming object detection:')
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        #print(detections)
        detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
        #print(detections)


    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
#cmap = plt.get_cmap('Vega20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print ('\nSaving images:')
# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print ("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    
    #############################
    lidar = np.load('/content/data_sample/point_cloud_npy/117.npy').reshape((-1, 4))[:, :3]
    height, width = img.shape[:2]
    depth_map = generate_dispariy_from_velo(lidar, height, width)
    depth_map=np.clip(depth_map, 0,255)
    depth_map=depth_map.astype(np.uint8)
    ###################################



    #kitti_img_size = 11*32
    kitti_img_size = 416
    # The amount of padding that was added
    #pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    #pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    pad_x = max(img.shape[0] - img.shape[1], 0) * (kitti_img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (kitti_img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = kitti_img_size - pad_y
    unpad_w = kitti_img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        print(type(detections))
        print(detections.size())
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
            # Rescale coordinates to original dimensions
            box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
            box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]) )
            y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
            x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))
            ########################################################
            indices=depth_map[y1:y1+box_h,x1:x1+box_w].nonzero()
            prediction_depth_box=depth_map[y1:y1+box_h,x1:x1+box_w]
            X=indices[1]+x1
            Y=indices[0]+y1
            #####################################################
            #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            #bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                    #edgecolor=color,
                                    #facecolor='none')
            colormap = cm.hot
            normalize = mcolors.Normalize(vmin=np.min(depth_map), vmax=np.max(depth_map))
            # Add the bbox to the plot
            ax.scatter(X, Y, c=prediction_depth_box[indices], s=1,cmap=colormap, norm=normalize )
            plt.show()
            #ax.add_patch(bbox)
            # Add label
            #plt.text(x1, y1-30, s=classes[int(cls_pred)]+' '+ str('%.4f'%cls_conf.item()), color='white', verticalalignment='top',
                    #bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    plt.close()
