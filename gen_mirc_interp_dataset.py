import scipy.io as sio
from matplotlib import pyplot as plt
import cv2
import CONSTS
import os

from mat2segmap import compute_seg_map

# ===========================
# Main
# ===========================
if __name__ == '__main__':

    visualization_flag = False
    if visualization_flag:
        fig = plt.figure()

    # creating folders:
    if not os.path.exists(CONSTS.MIRC_IMG_DIR):
        os.mkdir(CONSTS.MIRC_IMG_DIR)
    if not os.path.exists(CONSTS.LABEL_DIR):
        os.mkdir(CONSTS.LABEL_DIR)

    names = ['BIKE', 'HORSE_HEAD', 'MAN_IN_SUIT', 'HUMAN_EYE']

    for object_name in names:

            mat_fname = CONSTS.MAT_FILE_DIR + '/' + object_name + "_interpretation_data.mat"

            # Read MAT file:
            mat_contents = sio.loadmat(mat_fname)

            num_mirc_imgs = len(mat_contents[object_name][0])

            # create dirname:
            dirname = CONSTS.MIRC_IMG_DIR + '/' + object_name
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            dirname = CONSTS.LABEL_DIR + '/' + object_name
            if not os.path.exists(dirname):
                os.mkdir(dirname)

            for mirc_index in range(num_mirc_imgs):

                mirc_struct = mat_contents[object_name][0][mirc_index]
                mirc_img = mirc_struct['mirc']
                interp = mirc_struct['human_interpretation']

                # Resize mirc image:
                resized_mirc_img = cv2.resize(mirc_img, (CONSTS.INTERP_MAP_SIZE, CONSTS.INTERP_MAP_SIZE))

                # Compute seg map:
                canvas = compute_seg_map(resized_mirc_img, interp)

                # save map:
                label_filename = CONSTS.LABEL_DIR + '/' + object_name + '/' + str(mirc_index) + '.png'
                cv2.imwrite(label_filename, canvas)

                # save mirc image:
                img_filename = CONSTS.MIRC_IMG_DIR + '/' + object_name + '/' + str(mirc_index) + '.png'
                cv2.imwrite(img_filename, mirc_img)

                # optional: plot mirc_img + seg map:
                if visualization_flag:
                    # visualize:
                    fig.clf()
                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(resized_mirc_img, cmap='gray')
                    ax1.set_title("mirc")
                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.imshow(canvas, cmap='jet')
                    ax2.set_title("human interpretation")
                    plt.savefig()

