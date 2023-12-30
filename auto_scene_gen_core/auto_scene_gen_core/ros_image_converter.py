import numpy as np
import sys
import sensor_msgs.msg as sensor_msgs

encoding_to_dtypes = {
	"rgb8":    (np.uint8,  3),
	"rgba8":   (np.uint8,  4),
	"rgb16":   (np.uint16, 3),
	"rgba16":  (np.uint16, 4),
	"bgr8":    (np.uint8,  3),
	"bgra8":   (np.uint8,  4),
	"bgr16":   (np.uint16, 3),
	"bgra16":  (np.uint16, 4),
	"mono8":   (np.uint8,  1),
	"mono16":  (np.uint16, 1),
	
    # for bayer image (based on cv_bridge.cpp)
	"bayer_rggb8":	(np.uint8,  1),
	"bayer_bggr8":	(np.uint8,  1),
	"bayer_gbrg8":	(np.uint8,  1),
	"bayer_grbg8":	(np.uint8,  1),
	"bayer_rggb16":	(np.uint16, 1),
	"bayer_bggr16":	(np.uint16, 1),
	"bayer_gbrg16":	(np.uint16, 1),
	"bayer_grbg16":	(np.uint16, 1),

    # OpenCV CvMat types
	"8UC1":    (np.uint8,   1),
	"8UC2":    (np.uint8,   2),
	"8UC3":    (np.uint8,   3),
	"8UC4":    (np.uint8,   4),
	"8SC1":    (np.int8,    1),
	"8SC2":    (np.int8,    2),
	"8SC3":    (np.int8,    3),
	"8SC4":    (np.int8,    4),
	"16UC1":   (np.uint16,   1),
	"16UC2":   (np.uint16,   2),
	"16UC3":   (np.uint16,   3),
	"16UC4":   (np.uint16,   4),
	"16SC1":   (np.int16,  1),
	"16SC2":   (np.int16,  2),
	"16SC3":   (np.int16,  3),
	"16SC4":   (np.int16,  4),
	"32SC1":   (np.int32,   1),
	"32SC2":   (np.int32,   2),
	"32SC3":   (np.int32,   3),
	"32SC4":   (np.int32,   4),
	"32FC1":   (np.float32, 1),
	"32FC2":   (np.float32, 2),
	"32FC3":   (np.float32, 3),
	"32FC4":   (np.float32, 4),
	"64FC1":   (np.float64, 1),
	"64FC2":   (np.float64, 2),
	"64FC3":   (np.float64, 3),
	"64FC4":   (np.float64, 4)
}


def image_msg_to_numpy(msg: sensor_msgs.Image, keep_as_3d_tensor: bool = False):
	"""Convert a sensor_msgs/Image message to a numpy array
	
	Args:
		- msg: The image message
		- keep_as_3d_tensor: Indicates if the returned array should be a 3D array (only applies to 1-channel images)
	
	Returns:
		The converted image as a numpy array
	"""
	if not msg.encoding in encoding_to_dtypes:
		raise TypeError(f'Unrecognized encoding {msg.encoding}')

	dtype_class, channels = encoding_to_dtypes[msg.encoding]
	dtype = np.dtype(dtype_class)
	dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
	shape = (msg.height, msg.width, channels)

	# msg.data field is an array.array object
	data = np.fromstring(msg.data.tobytes(), dtype=dtype).reshape(shape)
	data.strides = (msg.step, dtype.itemsize * channels, dtype.itemsize)

	if channels == 1 and not keep_as_3d_tensor:
		data = data[...,0]

	return data