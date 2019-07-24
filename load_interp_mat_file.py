from os.path import dirname, join as pjoin
import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import cv2

mat_fname = "BIKE_interpretation_data.mat"
mat_contents = sio.loadmat(mat_fname)


sorted(mat_contents.keys())

mirc_struct = mat_contents['BIKE'][0][23]
img = mirc_struct['mirc']
interp = mirc_struct['human_interpretation']
points = interp[0][0]
contours = interp[0][1]
regions = interp[0][2]

# run over contours, e.g., :
resized_mirc = cv2.resize(img, (108,108))
c2 = contours[0][4]
#c2 = c2.reshape([c2.shape[0],1,c2.shape[1]])
cv2.drawContours(resized_mirc, [c2.astype(int)], 0, (0,255,0), 1)

#canvas = np.zeros_like(img)



#c = c2[43][0,1]

# visualize:
plt.figure()
plt.imshow(resized_mirc, cmap='gray')
plt.show()
