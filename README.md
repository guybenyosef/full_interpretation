# FullInterpretationDNNs


#### ‘Full interpretation of minimal images’ – human interpretation data

The ‘raw_data.zip’ file contains minimal image examples (as described in Sec. 5 in the main paper), and their
human interpretation data (referred below as annotations). The annotations for each category are stored in a
different MAT file. Each annotated image is represented as a MATLAB struct element with two fields: 'mirc',
which is the MIRC image and 'human_interpretation', which contains annotations for the MIRC image.

The human_interpretation field has three cells. The first is a list (represented as cells) of annotated [y,x] points.
The second cell is the list of contours and the third is a list of regions. Each contour is a matrix of size n by 2,
where n is the number of sampled [y,x] points in the contour. Each region is stored as a vector of [top row,
bottom row, left column, right column].

The file contains also a short MATLAB script for plotting image interpretation on screen.