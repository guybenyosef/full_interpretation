import os
import cv2
import sys
import argparse
import CONSTS
import random
import glob
import numpy as np

from utils.data_utils import make_dataset, make_dataset_txtfile

if os.path.exists(CONSTS.SELECTIVE_SEARCH_DIR):
    sys.path.insert(1, CONSTS.SELECTIVE_SEARCH_DIR)
    import selective_search

def argument_parser():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-m', '--mode', default='selective', help='selective or sliding. Default is selective.', type=str)
    parser.add_argument('-i', '--input_path', default='/Users/gby/data/minimal_images/negatives/nonhorse_large/1/', help='', type=str)
    parser.add_argument('-o', '--output_path', default='/Users/gby/data/minimal_images/negatives/nonhorse/1/', help='', type=str)
    parser.add_argument('-lm', '--limit', default=np.Inf, help='', type=int)
    parser.add_argument('-s', '--minimalImage_size', default=30, help='', type=int)
    parser.add_argument('-ns', '--num_sets', default=3, help='', type=int)

    return parser.parse_args()

def write_window_to_file(window, minimalImage_size, detection_indx, output_path, raw_name):
    # crop and save:
    window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    window = cv2.resize(window, (minimalImage_size, minimalImage_size))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    outputfilename = os.path.join(output_path, "{}_id{:07}.png".format(raw_name, detection_indx))
    cv2.imwrite(outputfilename, window)
    # print msg:
    if detection_indx > 0 and detection_indx % 1000 == 0:
        print("Detected windows.. {}".format(detection_indx))


def selective_search_detection(input_examples, minimalImage_size, output_path, limit=np.Inf):

    detection_indx = 0
    chunck_size = min(1000, limit // 1000)
    if chunck_size == 0:
        chunck_size = limit

    for img_path in input_examples:
        if os.path.exists(img_path):
            # read the image and define the stepSize and window size (width,height)
            print("processing {}".format(img_path))
            image = cv2.imread(img_path)  # your image path
            raw_name = os.path.splitext(os.path.basename(img_path))[0]

            boxes = selective_search.selective_search(image, mode='fast')
            random.shuffle(boxes)
            if limit < 10001:
                limit_per_image = 100
                boxes = boxes[:limit_per_image]

            for x1, y1, x2, y2 in boxes:
                window = image[x1:x2, y1:y2, :]

                if window.size > 0:
                    chunck_indx = detection_indx // chunck_size
                    write_window_to_file(window, minimalImage_size, detection_indx, os.path.join(output_path, str(chunck_indx)), raw_name)
                    detection_indx += 1
                    if detection_indx == limit:
                        return
                # draw window on image:
                # cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2)  # draw rectangle on image
                # plt.imshow(np.array(tmp).astype('uint8'))
                # plt.show()

    print("Total {} detected windows were saved to {}!".format(detection_indx, output_path))


def sliding_window_detection(input_examples, minimalImage_size, output_path, limit):

    detection_indx = 0

    for img_path in input_examples:
        # read the image and define the stepSize and window size (width,height)
        image = cv2.imread(img_path)  # your image path

        for window_size in range(minimalImage_size, 200, 10):

            for stepSize in range(minimalImage_size, minimalImage_size*2, 5):

                (w_width, w_height) = (window_size, window_size)  # window size

                for x in range(0, image.shape[1] - w_width, stepSize):

                    for y in range(0, image.shape[0] - w_height, stepSize):

                        window = image[x:x + w_width, y:y + w_height, :]

                        # classify content of the window with your classifier and
                        if window.size > 0:
                            write_window_to_file(window, minimalImage_size, detection_indx)
                            detection_indx += 1
                            if detection_indx > limit:
                                return
                        # draw window on image
                        # cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2)  # draw rectangle on image
                        # plt.imshow(np.array(tmp).astype('uint8'))
                        # plt.show()
    print("Total {} detected windows were saved to {}!".format(detection_indx, output_path))


def gen_data(mode, input_filenames, output_path, num_sets, minimalImage_size, limit):

    random.shuffle(input_filenames)

    set_size = len(input_filenames) // num_sets
    if set_size == 0:
        num_sets, set_size = 1, 1

    for set_indx in range(num_sets):
        set_output_path = os.path.join(output_path, str(set_indx))
        if not os.path.exists(set_output_path):
            os.makedirs(set_output_path)

        files_subset = input_filenames[set_indx * set_size : (set_indx + 1) * set_size]

        if mode == 'selective':
            selective_search_detection(files_subset, minimalImage_size, set_output_path, limit)
        elif mode == 'sliding':
            sliding_window_detection(files_subset, minimalImage_size, set_output_path, limit)
        else:
            os.error("incorrect mode name")


def get_voc_classification_filenames(voc_folder_path, category='horse'):

    voc_classification_names_folder = os.path.join(voc_folder_path, 'ImageSets/Main/')
    voc_classification_images_folder = os.path.join(voc_folder_path, 'JPEGImages/')

    all_voc_classification_names = make_dataset_txtfile(os.path.join(voc_classification_names_folder, 'train.txt')) + \
                                   make_dataset_txtfile(os.path.join(voc_classification_names_folder, 'val.txt')) + \
                                   make_dataset_txtfile(os.path.join(voc_classification_names_folder, 'trainval.txt'))
    category_voc_classification_names = make_dataset_txtfile(os.path.join(voc_classification_names_folder, category + '_train.txt')) + \
                                   make_dataset_txtfile(os.path.join(voc_classification_names_folder, category + '_val.txt')) + \
                                   make_dataset_txtfile(os.path.join(voc_classification_names_folder, category + '_trainval.txt'))

    all_but_category_voc_classification_names = list(set(all_voc_classification_names) - set(category_voc_classification_names))

    # add full path
    all_but_category_voc_classification_names = [os.path.join(voc_classification_images_folder, filename + '.jpg') for filename in all_but_category_voc_classification_names]

    return all_but_category_voc_classification_names


# ===========================
# Main
# ===========================
if __name__ == '__main__':

    args = argument_parser()

    input_path = args.input_path
    output_path = args.output_path

    if input_path == 'voc_horse':
        # get all nonhorse voc files:
        input_filenames = get_voc_classification_filenames(voc_folder_path=CONSTS.VOC_DIR, category='horse')
    elif input_path == 'voc_mis':
        # get all nonhorse voc files:
        input_filenames = get_voc_classification_filenames(voc_folder_path=CONSTS.VOC_DIR, category='person')
    else:
        input_filenames = make_dataset(dir=input_path, ext='jpg')

    gen_data(mode=args.mode, input_filenames=input_filenames, output_path=output_path,
            num_sets=args.num_sets, minimalImage_size=args.minimalImage_size, limit=args.limit)






    # # Filter box proposals
    # # Feel free to change parameters
    # boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=80)
    #
    # # draw rectangles on the original image
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(image)
    # for x1, y1, x2, y2 in boxes_filter:
    #     bbox = mpatches.Rectangle(
    #         (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='red', linewidth=1)
    #     ax.add_patch(bbox)
    #
    # plt.axis('off')
    # plt.show()

