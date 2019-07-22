from os.path import dirname, join as pjoin
import scipy.io as sio
from matplotlib import pyplot as plt

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
c2 = contours[0][2]

# visualize:
plt.figure()
plt.imshow(img, cmap='gray')
plt.show()
