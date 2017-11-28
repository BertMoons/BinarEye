from keras.preprocessing.image import transform_matrix_offset_center, apply_transform
import keras.backend as K
import numpy as np

def random_transform(x, rotation_range=0, height_shift_range = 0,width_shift_range=0,shear_range=0,zoom_range=(1,1)):
	# x is a single image, so it doesn't have image number at index 0
	img_row_index = 1
	img_col_index = 0
	img_channel_index = 2
	fill_mode = 'nearest'
	cval = 0
	K.set_image_format = 'channels_last'


	# use composition of homographies to generate final transform that
	# needs to be applied
	need_transform = False

	# Rotation
	if rotation_range:
		theta = np.pi / 180 * np.random.uniform(-rotation_range,
							rotation_range)
		need_transform = True
	else:
		theta = 0

	# Shift in height
	if height_shift_range:
		tx = np.random.uniform(-height_shift_range,
					   height_shift_range) * x.shape[img_row_index]
		need_transform = True
	else:
		tx = 0

	# Shift in width
	if width_shift_range:
		ty = np.random.uniform(-width_shift_range,
				   width_shift_range) * x.shape[img_col_index]
		need_transform = True
	else:
		ty = 0

	# Shear
	if shear_range:
		shear = np.random.uniform(-shear_range, shear_range)
		need_transform = True
	else:
		shear = 0

	# Zoom
	if zoom_range[0] == 1 and zoom_range[1] == 1:
		zx, zy = 1, 1
	else:
		zx, zy = np.random.uniform(zoom_range[0],
					   zoom_range[1], 2)
		need_transform = True

	if need_transform:
		rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
					[np.sin(theta), np.cos(theta), 0],
					[0, 0, 1]])

		translation_matrix = np.array([[1, 0, tx],
					   [0, 1, ty],
					   [0, 0, 1]])

		shear_matrix = np.array([[1, -np.sin(shear), 0],
					 [0, np.cos(shear), 0],
					 [0, 0, 1]])

		zoom_matrix = np.array([[zx, 0, 0],
					[0, zy, 0],
					[0, 0, 1]])

		transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
							translation_matrix),
						 	shear_matrix), zoom_matrix)

		h, w = x.shape[img_row_index], x.shape[img_col_index]
		transform_matrix = transform_matrix_offset_center(transform_matrix,
								  h, w)
		x = apply_transform(x, transform_matrix, img_channel_index,
							fill_mode=fill_mode, cval=cval)

	return x


def augment_images(x,rotation_range=0, height_shift_range = 0,width_shift_range=0,shear_range=0,zoom_range=(1,1)):
	y = np.array([]).reshape(0,32,32,3)
	for i in range(0,x.shape[0]):
		y_ = random_transform(x[i],rotation_range=rotation_range, height_shift_range = height_shift_range,width_shift_range=width_shift_range,shear_range=shear_range,zoom_range=zoom_range)
		y_ = np.expand_dims(y_,axis=0)		
		y = np.concatenate((y,y_),axis=0)

	return y


def adapt_to_binareye(x, filters=64):
	x = np.subtract(np.multiply(2./255.,x),1.)
	# Number of feature maps
	num_maps = np.floor(256/(3*256/filters)) # 256/3 input channels
	# Quantize
	s = x / np.abs(x)
	x=(2*(s*np.ceil(np.abs(x)*num_maps/2))-s*1).astype('float32')
	return x
        
