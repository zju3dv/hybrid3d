import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

def colored_data(x, cmap='viridis', d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(x)
    if d_max is None:
        d_max = np.max(x)
    print(np.min(x), np.max(x))
    x_relative = (x - d_min) / (d_max - d_min)
    cmap_ = plt.cm.get_cmap(cmap)
    return (255 * cmap_(x_relative)[:,:,:3]).astype(np.uint8) # H, W, C

rgb1 = np.load('debug/v8/rgb_7-scenes-redkitchen_0.npy')[0]
rgb2 = np.load('debug/v8/rgb_7-scenes-redkitchen_4.npy')[6]
desc1 = np.load('debug/v8/desc_7-scenes-redkitchen_0.npy')[0]
desc2 = np.load('debug/v8/desc_7-scenes-redkitchen_4.npy')[6]
# rgb1 = 255 - cv2.imread('data/3dmatch/rgbd/7-scenes-redkitchen/seq-01/frame-000000.color.png').transpose(2, 0, 1)
# rgb2 = 255 - cv2.imread('data/3dmatch/rgbd/7-scenes-redkitchen/seq-01/frame-000252.color.png').transpose(2, 0, 1)
# rgb2 = np.load('debug/v7/rgb_7-scenes-redkitchen_6.npy')[0]
# desc1 = np.load('/home/ybbbbt/Developer/r2d2/desc1.npy')[0]
# desc2 = np.load('/home/ybbbbt/Developer/r2d2/desc2.npy')[0]
# desc1 = np.load('/home/ybbbbt/Developer/d2-net/desc1.npy')[0]
# desc2 = np.load('/home/ybbbbt/Developer/d2-net/desc2.npy')[0]
print(desc1.shape)

def upscale(desc, size):
    desc = torch.nn.functional.interpolate(torch.from_numpy(desc[None, ...]), size=size, mode='bilinear', align_corners=True)
    print(desc.shape)
    desc = torch.nn.functional.normalize(desc, dim=1)
    return desc.numpy()[0]

if desc1.shape[1] < rgb1.shape[1]:
    desc1 = upscale(desc1, rgb1.shape[1:3])
    desc2 = upscale(desc2, rgb2.shape[1:3])
# xyz1 = np.load('debug/plain_fix_w/xyz_7-scenes-redkitchen_0.npy')[0]
# xyz2 = np.load('debug/plain_fix_w/xyz_7-scenes-redkitchen_10.npy')[0]

kpt = (223, 270) #x y

# rgb1 = rgb_np[0]
# rgb2 = rgb_np[4]
# desc1 = desc_np[0]
# desc2 = desc_np[4]

rgb1 = (rgb1.transpose(1, 2, 0) * 255).astype(np.uint8).copy()
rgb2 = (rgb2.transpose(1, 2, 0) * 255).astype(np.uint8).copy()

rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)
rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR)

rgb1_marker = rgb1
blend = rgb2

# coord_color = colored_data(np.linalg.norm(xyz1, axis=0))
# # print(np.linalg.norm(xyz1, axis=0).shape)
# import open3d as o3d
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz1.reshape(3, -1).transpose(1, 0))
# o3d.io.write_point_cloud('test.ply', pcd)

def mouse_callback(event, x, y, flags, param):
    global mouseX, mouseY
    global rgb1, rgb2, rgb1_marker, blend
    if event == cv2.EVENT_LBUTTONDBLCLK:
        rgb1_marker = cv2.drawMarker(rgb1.copy(), (x, y), color=(0, 255, 0), thickness=2)
        mouseX, mouseY = x,y
        desc1_kpt = desc1[:, y, x]
        # dists = np.linalg.norm(desc2 - desc1_kpt[:, None, None], axis=0)
        # dists_colored = colored_data(dists, cmap='coolwarm', d_min=0.3, d_max=0.8)
        # sim = np.matmul(desc2, desc1_kpt)
        # sim = np.linalg.norm(desc2 - desc1_kpt[:, None, None], axis=0)
        sim = np.einsum('ijk,i->jk', desc2, desc1_kpt, optimize = True)
        # sim_colored = colored_data(sim, cmap='coolwarm', d_min=0.3, d_max=0.8)
        sim_colored = colored_data(sim, cmap='coolwarm', d_min=0.5, d_max=1)
        # dists_colored = colored_data(dists, cmap='coolwarm')
        # dists_colored = colored_data(dists, cmap='coolwarm', d_min=1000, d_max=5000)
        # blend = (0.5 * dists_colored + 0.5 * rgb2).astype(np.uint8)
        blend = (0.5 * sim_colored + 0.5 * rgb2).astype(np.uint8)
        max_pt = np.unravel_index(np.argmax(sim), sim.shape)
        print(max_pt)
        blend = cv2.drawMarker(blend.copy(), (max_pt[1], max_pt[0]), color=(0, 255, 0), thickness=2)

cv2.namedWindow('image')
cv2.setMouseCallback('image',mouse_callback)

while(1):
    cv2.imshow('image',rgb1_marker)
    cv2.imshow('blend',blend)
    # cv2.imshow('coord',coord_color)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(mouseX, mouseY)



# plt.figure(1)

# # print(blend)
# plt.imshow(blend)
# # plt.imshow(dists, cmap='viridis')
# # plt.colorbar()
# plt.figure(2)
# plt.imshow(rgb1)
# plt.show()
