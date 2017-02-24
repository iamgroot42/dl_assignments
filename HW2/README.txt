Location of Dataset: 192.168.22.72/IBSR_540K
Dataset Description:
The IBSR dataset consists of brain volumes of size 256x128x256 taken from 18 different subjects. For the purpose of this assignment you are given
volumes extracted from 10 of these subjects. The volume are of size 13x13x13 and are from three regions of the Brain - CerebroSpinal Fluid(CSF),
White Matter(WM) and Grey Matter(GM)

The training data is given in train_540k.mat and has a shape of 2197x540000 where each 13x13x13 volume is flattened and laid out as a column

The labels are given as label_csf.mat, label_gm.mat, label_wm.mat and are binary arrays of shape 1x540000. The data is interpreted as: If data index idx belongs to class 'x' then label_x.mat would contain a 1 at position idx. For instance data index 10 belongs to class 'csf', hence, label_csf contains a 1 at position 10

A python script to read the data is provided in the data folder