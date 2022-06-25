from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio



'''
=================== Definitions ===================
num_pixels: number of pixels in input image
num_channels: number of color channels in input image (for RGB input image, num_channels=3)
num_clusters: number of clusters/centroids to reduce pixel colors to/number of 
	clusters/centroids in compressed output image
'''


# ================= Functions =================
def get_centroid_indices(X, centroids):
	'''
	For each pixel in input image X, find closest color centroid to that pixel's color values.

	Arguments
	---------
	X: array-like
		normalized input image of shape (num_pixels, num_channels)
	centroids: array-like
		color centroids of shape (num_clusters, num_channels) 

	Returns
	-------
	idx: array-like
		ndarray of shape (num_pixels, 1) indicating, for each pixel in input image,
		index of closest color centroid to that pixel
	'''

	K = centroids.shape[0]								# number of clusters/colors			
	idx = np.zeros((X.shape[0], 1))						# indices array of shape (num_pixels, 1)
	d = np.zeros((X.shape[0], centroids.shape[0]))		# color centroid distance array of shape (num_pixels, num_clusters)

	for i in range(X.shape[0]):							# for each pixel in input image
		for j in range(centroids.shape[0]):				# for each K cluster/color

			diff = np.subtract(X[i], centroids[j])		# compute distance between ith pixel color and jth color centroid
			d[i,j] = np.sum(np.square(diff))			# square distance and sum across color channels

			iy = np.where(d[i]==np.min(d[i]))[0][0]		# for ith pixel, find index of cluster/color having smallest distance to ith pixel
			idx[i] = iy									# add this index to index array
			
	return idx


def get_centroid_values(X, idx, K):
	'''
	Compute color centroid pixel values for each cluster.

	Arguments
	---------
	X: array-like
		normalized input image of shape (num_pixels, num_channels)
	idx: array-like
		ndarray of shape (num_pixels, 1) indicating, for each pixel in input image,
		index of closest color centroid to that pixel
	K: int
		number of clusters/colors to reduce pixel values to
	
	Returns
	-------
	centroids: array-like
		color centroid pixel values for each cluster; of shape (num_clusters, num_channels)
	'''

	m = X.shape[0]						# num_pixels
	n = X.shape[1]						# num_channels
	centroids = np.zeros((K, n))		# shape of (num_clusters, num_channels)

	for i in range(K):					# for each cluster
		indices = np.where(idx==i)		# get indices where idx == the ith cluster 					

		# indices is a tuple of len 2; indices[0] is ndarray of rank 1 arrays
		num = len(indices[0])					

		# if num is 0, avoid division by 0
		if num == 0:
			num = 1

		# X[indices[0]]	selects pixels corresponding to ith cluster
		# compute centroid for ith cluster by averaging the values of all pixels corresponding to ith cluster
		centroids[i] = (1/num) * np.sum(X[indices[0]], axis=0)		

	return centroids


def apply_k_means(X, initial_centroids, iters):
	'''
	Apply k-means clustering to compute color centroids for each cluster.

	Arguments
	---------
	X: array-like
		normalized input image of shape (num_pixels, num_channels)
	initial_centroids: array-like
		initial color centroids of shape (num_clusters, num_channels) 
	iters: int
		number of iterations to perform image compression

	Returns
	-------
	centroids: array-like
		color centroid pixel values for each cluster; of shape (num_clusters, num_channels)
	idx: array-like
		ndarray of shape (num_pixels, 1) indicating, for each pixel in input image,
		index of closest color centroid to that pixel
	'''

	m = X.shape[0]							# num_pixels
	n = X.shape[1]							# num_channels
	K = initial_centroids.shape[0]			# num_clusters
	centroids = initial_centroids			
	idx = np.zeros((m, 1))

	# starting with initial_centroids, iteratively optimize for iters to obtain best centroids
	for i in range(iters):
		idx = get_centroid_indices(X, centroids)		# for each pixel in image X, find index of closest color centroid
		centroids = get_centroid_values(X, idx, K)		# for each cluster, compute color centroid pixel values 

	return centroids, idx


def initialize_centroids(X, K):
	'''
	Initialize K color centroids. Color centroids consist of pixel color value(s).
	For RGB input image, each color centroid consists of R, G, and B pixel values.

	Arguments
	---------
	X: array-like 
		normalized input image of shape (num_pixels, num_channels)
	K: int
		number of clusters/colors to reduce pixel values to

	Returns
	-------
	centroids: array-like
		initial color centroids of shape (num_clusters, num_channels) 
	'''

	centroids = np.zeros((K, X.shape[1]))				# shape of (num_clusters, num_channels)
	randidx = np.random.permutation(X.shape[0])			# row vector formed by randomly permuting num_pixels in X 
	centroids = X[randidx[:K]]							# get K initial color centoids by selecting K random pixel values

	return centroids



# ================= Script =================
# ==========================================
# ==========================================

img = imageio.imread('image.jpg')								# Load input image as array of shape (height, width, num_channels)
X = img / 255													# Normalize pixel values
X = np.reshape(X, (img.shape[0]*img.shape[1], 3))				# Reshape into 2D array of shape (num_pixels, num_channels)

# Settings
K = 16															# Number of clusters/colors to reduce pixel values to
ITERS = 10														# Number of iterations to perform image compression

initial_centroids = initialize_centroids(X, K)					# Initialize color centroids to be later optimized
centroids, idx = apply_k_means(X, initial_centroids, ITERS)		# get color centroids for each cluster and index of closest centroid for each pixel
idx = idx.astype('int')											# idx comprises float64; to use idx as indices into X, convert to ints

'''
idx essentially comprises all of the pixel color values of the compressed image, 
but in the form of indices that each indicate the color centroid/cluster to
which a corresponding pixel is mapped to. idx is used to index into centroids, 
which comprises the pixel values (R/G/B coordinates for an RGB input image) 
for each centroid/cluster. This indexing performs broadcasting - idx is an array
of shape (num_pixels, 1), while centroids is an array of shape (num_clusters, num_channels).
centroids[idx] produces an array - which when properly reshaped produces the compressed
output image - of shape (num_pixels, 1, num_channels)
'''
X_recovered = centroids[idx]											# construct compressed output image
X_recovered *= 255														# rescale pixel values to within range [0, 255]
X_recovered = X_recovered.astype('uint8')								# convert pixel values from float64 to int8
X_recovered = np.reshape(X_recovered, (img.shape[0], img.shape[1], 3))	# reshape from (num_pixels, num_channels) to (height, width, num_channels)
X_recovered = Image.fromarray(X_recovered)								# convert array to PIL Image type for plt.show()

# Show original input image and compressed output image
fig, axs = plt.subplots(1, 2)
axs[0].imshow(img)
axs[0].set_title('Original Image')
axs[1].imshow(X_recovered)
axs[1].set_title('Compressed Image')
plt.show()

X_recovered.save('image_compressed.jpg')						# Save compressed output image
