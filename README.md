# Full Interpretation DNNs


### Full interpretation of minimal images – Detailed human identification of object parts

#### Raw matlab data:

The folder [raw_data](https://www.dropbox.com/sh/faktmyhspckyjbj/AAB-xPG_dVJKWScWDaAr6WaNa?dl=0) contains minimal image examples 
(also referred as MIRCs), and their
human interpretation data (referred below as annotations). 
The annotations for each category are stored in a
different MAT file. Each annotated image is represented as a MATLAB struct element with two fields:
***'mirc'***, which is the MIRC image 
and ***'human_interpretation'***, which contains annotations for the MIRC image.

The human_interpretation field has three cells. 
The first is a list (represented as cells) of annotated *[y,x]* points.
The second cell is the list of contours and the third is a list of regions. 
Each contour is a matrix of size n by 2,
where n is the number of sampled *[y,x]* points in the contour. 
Each region is stored as a vector of *[top row,
bottom row, left column, right column]*.

The zip file contains also a short MATLAB script for plotting interpretations.

#### Python installation

Extract raw_data.zip and update the file CONSTS.py to your current folder locations.

###### Visualize interpretation examples:
To visualize a mirc image and its interpretation run, e.g.,     
`$python mat2segamp.py -n HORSE_HEAD -i 13`     
where -n is input arg for mirc object name, and -i is the mirc image index.


###### Install full intepretation dataset:  
To create the full dataset of mirc image and their interpretation as segmentation maps:
Update the file CONSTS.py to your current folder locations, and then run:    
`$python gen_mirc_interp_dataset.py`    

This should create a ~12M size folder containing mirc images as well as their segmentation maps. 


#### Paper
If you use this dataset please cite the following 
[paper](https://www.researchgate.net/publication/320921911_Full_interpretation_of_minimal_images):    

@article{ben2018full,   
  title={Full interpretation of minimal images},    
  author={Ben-Yosef, Guy and Assif, Liav and Ullman, Shimon},   
  journal={Cognition},  
  volume={171},  
  pages={65--84},       
  year={2018},  
  publisher={Elsevier}  



