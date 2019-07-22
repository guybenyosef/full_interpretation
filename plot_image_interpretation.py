# % -- Visualize image interpretation ---
# % ---------------------------------------------------------------------------------
# % Description: plot detailed image interpreation
# % Type:        Visualization procedure
# % INPUT:        figure_id: a natural number (refers to MATLAB's figure object for the plot)
# %               img: an image in 108x108 pixel size.
# %               interpretation_set: a MATLAB strcut of points, contours, and regions
# %
# %  Run, e.g.:
# %
# % >> plot_image_interpretation(1,imresize(HORSE_HEAD(36).mirc,[108,108]),HORSE_HEAD(36).human_interpretation,true);
# %
# % Guy Ben-Yosef & Liav Assif, 2017, gby@csail.mit.edu
# % ---------------------------------------------------------------------------------  %

from matplotlib import pyplot as plt

LWIDTH = 3
# LWIDTH = 8; % for publications

show_img_flag = True

def plot_image_interpretation(figure_id,img,interpretation_set):


    if show_img_flag:
        fig = plt.figure(figure_id)
        fig.clf()
        fig.imshow(img)
