import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import CONSTS

def argument_parser():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-n', '--object_name', default='BIKE', help='choose between \'BIKE\',\'HORSE_HEAD\',\'MAN_IN_SUIT\', and \'HUMAN_EYE\' object names', type=str)
    parser.add_argument('-i', '--mirc_index', default=23, help='MIRC index', type=int)

    return parser.parse_args()

def compute_seg_map(resized_mirc_img, interp):
    points = interp[0][0]
    contours = interp[0][1]
    regions = interp[0][2] # ignore

    if len(points):
        contours = np.concatenate((contours, points), axis=1)

    canvas = np.zeros_like(resized_mirc_img)

    # run over contours and add to canvas:
    for cntr_indx in range(len(contours[0])):
        c = contours[0][cntr_indx]
        for k in range(len(c)):
            row = int(c[k, 0] - 1)
            col = int(c[k, 1] - 1)
            if row < CONSTS.INTERP_MAP_SIZE and col < CONSTS.INTERP_MAP_SIZE:
                #print("r: %d,  c: %d" % (row, col))
                canvas[row, col] = cntr_indx + 1

    return canvas


# ===========================
# Main
# ===========================
if __name__ == '__main__':

    # Parse args:
    # -----------
    args = argument_parser()

    object_name = args.object_name
    mat_fname = CONSTS.MAT_FILE_DIR + '/' + object_name + "_interpretation_data.mat"
    mirc_index = args.mirc_index

    # Read MAT file:
    mat_contents = sio.loadmat(mat_fname)
    #
    mirc_struct = mat_contents[object_name][0][mirc_index]
    mirc_img = mirc_struct['mirc']
    interp = mirc_struct['human_interpretation']

    # Resize mirc image:
    resized_mirc_img = cv2.resize(mirc_img, (CONSTS.INTERP_MAP_SIZE, CONSTS.INTERP_MAP_SIZE))

    # Compute seg map:
    canvas = compute_seg_map(resized_mirc_img, interp)

    # plot seg map:
    visualization_flag = True
    if visualization_flag:
        # visualize:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(resized_mirc_img, cmap='gray')
        ax1.set_title("mirc")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(canvas, cmap='jet')
        ax2.set_title("human interpretation")
        plt.show()

    # save map:
    label_filename = CONSTS.LABEL_DIR + '/' + args.object_name + '/' + str(args.mirc_index) + '.png'
    cv2.imwrite(label_filename, canvas)