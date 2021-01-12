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
c2 = contours[0][2]
x = c2[:,0]
y = c2[:,1]
c2_ = c2.copy()
c2_[:, 0] = y#y
c2_[:, 1] = x#108-x
#c2 = c2.reshape([c2.shape[0],1,c2.shape[1]])
cv2.drawContours(resized_mirc, [c2_.astype(int)], 0, (255,0,255), 1)

canvas = np.zeros_like(resized_mirc)
for k in range(len(c2)):
    canvas[c2[k,0],c2[k,1]]=1


#resized_mirc = canvas

#c = c2[43][0,1]

# visualize:
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(resized_mirc, cmap='gray')
fig.add_subplot(1,2,2)
plt.imshow(canvas, cmap='gray')
plt.show()
