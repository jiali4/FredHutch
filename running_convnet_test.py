import h5py
import tifffile as tiff
from keras.backend.common import _UID_PREFIXES

from cnn_functions import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes, segment_nuclei, segment_cytoplasm, dice_jaccard_indices
from model_zoo import sparse_bn_feature_net_61x61 as cyto_fn
from model_zoo import sparse_bn_feature_net_61x61 as nuclear_fn

import os
import numpy as np

direc_name = '/home/ubuntu/DeepCell/validation_data/HeLa/'
data_location = os.path.join(direc_name, 'RawImages')

cyto_location = os.path.join(direc_name, 'Cytoplasm')
nuclear_location = os.path.join(direc_name, 'Nuclear')
mask_location = os.path.join(direc_name, 'Masks')

cyto_channel_names = ['phase', 'farred']
nuclear_channel_names = ['farred']

trained_network_cyto_directory = "/home/ubuntu/DeepCell/trained_networks/HeLa/"
trained_network_nuclear_directory = "/home/ubuntu/DeepCell/trained_networks/Nuclear/"

cyto_prefix = "2017-06-21_HeLa_all_61x61_bn_feature_net_61x61_"
nuclear_prefix = "2016-07-12_nuclei_all_61x61_bn_feature_net_61x61_"

win_cyto = 30
win_nuclear = 30

image_size_x, image_size_y = get_image_sizes(data_location, nuclear_channel_names)
image_size_x /= 2
image_size_y /= 2

list_of_cyto_weights = []
for j in xrange(5):
	cyto_weights = os.path.join(trained_network_cyto_directory,  cyto_prefix + str(j) + ".h5")
	list_of_cyto_weights += [cyto_weights]

list_of_nuclear_weights = []
for j in xrange(5):
	nuclear_weights = os.path.join(trained_network_nuclear_directory,  nuclear_prefix + str(j) + ".h5")
	list_of_nuclear_weights += [nuclear_weights]

cytoplasm_predictions = run_models_on_directory(data_location, cyto_channel_names, cyto_location, model_fn = cyto_fn, 
	list_of_weights = list_of_cyto_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
	win_x = win_cyto, win_y = win_cyto, std = False, split = False)

nuclear_predictions = run_models_on_directory(data_location, nuclear_channel_names, nuclear_location, model_fn = nuclear_fn, 
	list_of_weights = list_of_nuclear_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
	win_x = win_nuclear, win_y = win_nuclear, std = False, split = False)

nuclear_masks = segment_nuclei(nuclear_predictions, mask_location = mask_location, threshold = 0.75, area_threshold = 100, solidity_threshold = 0.75, eccentricity_threshold = 0.95)
cytoplasm_masks = segment_cytoplasm(cytoplasm_predictions, nuclear_masks = nuclear_masks, mask_location = mask_location, smoothing = 1, num_iters = 120)

direc_val = os.path.join(direc_name, 'Validation')
imglist_val = nikon_getfiles(direc_val, 'validation_interior')

val_name = os.path.join(direc_val, imglist_val[0]) 
val = get_image(val_name)
val = val[win_cyto:-win_cyto,win_cyto:-win_cyto]
cyto = cytoplasm_masks[0,win_cyto:-win_cyto,win_cyto:-win_cyto]
nuc = nuclear_masks[0,win_cyto:-win_cyto,win_cyto:-win_cyto]

dice_jaccard_indices(cyto, val, nuc)