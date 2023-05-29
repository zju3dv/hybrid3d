import numpy as np
import os
import sys
import subprocess
import re

def get_fmr(path, inlier_ratio_th, inlier_dist_th):
    text_result = subprocess.check_output(
        ['python', 'evaluation/registration_3dmatch.py',
        '5000', 'data/3dmatch/geometric_registration_adaptive',
        f'{path}', str(inlier_ratio_th), str(inlier_dist_th)]
    ).decode(sys.stdout.encoding)
    # print(text_result)
    return re.findall("All 8 scene, average recall: (\d+\.\d+)", text_result)[0]

if __name__ == '__main__':
    path = sys.argv[1]

    inlier_ratio_th_base = 0.05
    inlier_ratio_th_list = list(np.arange(0, 0.21, 0.01))

    inlier_dist_th_base = 0.1
    inlier_dist_th_list = list(np.arange(0, 0.21, 0.01))

    print('# FMR, tau_1, tau_2')

    for tau_1 in inlier_ratio_th_list:
        fmr = get_fmr(path, tau_1, inlier_dist_th_base)
        print('{}, {}, {}'.format(tau_1, inlier_dist_th_base, fmr))
        
    for tau_2 in inlier_dist_th_list:
        fmr = get_fmr(path, inlier_ratio_th_base, tau_2)
        print('{}, {}, {}'.format(inlier_ratio_th_base, tau_2, fmr))
