import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2


if __name__ == "__main__":
    # desc
    if False:
        desc = np.load('/home/ybbbbt/Developer/hybrid_feature/debug/256_256_subset_601_dump_for_vis/desc_8_1.npy')
        desc = desc[3]
        desc = desc.max(axis=0)

        plt.imshow(desc, cmap='viridis')
        plt.colorbar()
        plt.imsave('desc.png', desc, cmap='viridis')
        plt.show()

        print(desc.shape)
        print(desc)

    # coord
    if True:
        coord = np.load('/home/ybbbbt/Developer/hybrid_feature/debug/256_256_subset_601_dump_for_vis/coord_8_1.npy')
        coord = coord[3]
        for i in range(60):
            for j in range(80):
                coord[1, i, j] -= i * 480 / 60
                coord[0, i, j] -= j * 640 / 80
        coord = np.linalg.norm(coord, axis=0)
        
        plt.imshow(coord, cmap='coolwarm')
        plt.colorbar()
        plt.imsave('coord.png', coord, cmap='coolwarm')
        plt.show()

