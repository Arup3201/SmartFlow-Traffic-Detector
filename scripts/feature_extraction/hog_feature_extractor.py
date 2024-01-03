import math
import numpy as np
from PIL import Image

class HistogramOrientedGradients:
    '''
    HOG class implements the histogram oriented gradient algorithm to extract features from an image.
    '''
    def __init__(self, img_dimension=(128, 64), pixels_per_cell=(8, 8), buckets=9, cells_per_block=(2, 2), channel_axis=None):
        '''
        Initialize the class with parameter values.

        args:
        `img_dimension`: tuple of 2 values, shape to which image should be resized before performing hog `(width, height)`
        `pixels_per_cell`: int, number of pixels taken in each cell horizontally and vertically
        `buckets`: int, number of groups or bins of orientations to generate the histogram
        `cells_per_block`: int, number of cells for which gradient normalization will be performed

        returns:
        None
        '''

        self.img_dimension = img_dimension
        self.pixels_per_cell = pixels_per_cell
        self.buckets = buckets
        self.cells_per_block = cells_per_block
        self.channel_axis = channel_axis
        self.multichannel = True if self.channel_axis is not None else False

    def _open_image(self, img_path):
        '''
        opens the image, converts it into gray scale and then resizes it according to the image dimension mentioned and finally converts it into numpy array of float32 data type.
        
        args:
        `img_path`: str, path to the image with respect to root directory

        returns:
        image after all tranformations
        '''
        img = Image.open(img_path)
        img = img.convert('L')
        img = img.resize(self.img_dimension)
        img = np.atleast_2d(img)
        img = img.astype(np.float32)

        return img
    
    def _get_gradient_channel(self, g_row, g_col, channel):
        g_row[0, :] = 0
        g_col[:, 0] = 0
        g_row[-1, :] = 0
        g_col[:, -1] = 0

        g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
        g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]

        return g_row, g_col
    

    def calculate_magnitude_angels(self, img, ep=1e-6):
        '''
        calculates magnitude and orientations of the image `img`.
        
        args:
        `img`: np.ndarray, array of shape self.img_dimension representation of the image

        returns:
        magnitudes and orientations of the image
        '''

        g_row = np.empty_like(img, dtype=np.float32)
        g_col = np.empty_like(img, dtype=np.float32)

        g_mag = np.empty_like(img, dtype=np.float32)
        g_ang = np.empty_like(img, dtype=np.float32)

        if self.multichannel:
            for ch_id in range(img.shape[self.channel_axis]):
                g_row, g_col = self._get_gradient_channel(g_row, g_col, img[:, :, ch_id])
        else:
            g_row, g_col = self._get_gradient_channel(g_row, g_col, img[:, :])

        
        g_mag = np.hypot(g_row, g_col)
        g_ang = np.rad2deg(np.arctan(g_row, g_col+ep)) % 180

        return g_mag, g_ang
    
    def generate_histogram(self, cell_orientations, cell_magnitudes):
        '''
        generates histogram for a cell given its orientation and magnitude. Calculates the step size and creates bins of frequency. Puts the magnitude according to each bins contribution.
        
        args:
        `cell_orientations`: numpy.ndarray, array with shape (self.pixels_per_cell, self.pixels_per_cell) having the orientation of each pixel
        `cell_magnitudes`: numpy.ndarray, array with shape (self.pixels_per_cell, self.pixels_per_cell) having the magnitude of each pixel

        returns:
        numpy.ndarray, histogram of shape (self.buckets, 1)
        '''
        matrix = np.zeros(shape=(self.buckets, 1))
        step_size = 180//self.buckets

        nr = self.pixels_per_cell[0]
        nc = self.pixels_per_cell[1]

        for r in range(nr):
            for c in range(nc):
                # Take the magnitude and angel
                theta = cell_orientations[r, c]
                magnitude = cell_magnitudes[r, c]

                # Find the jth and j+1th bin in which the orientation falls
                j_bin = int(theta / step_size)
                j_1_bin = int(theta / step_size) % self.buckets # When j = 8 next is j+1 = 0

                # Calculate the jth and j+1th bins contribution to the orientation
                Vj = ((j_bin * step_size + step_size) - theta) / step_size # (40-36)/20
                Vj_1 = (theta - (j_bin * step_size)) / step_size # (36-20)/20

                matrix[j_bin] = Vj * magnitude
                matrix[j_1_bin] = Vj_1 * magnitude

        return matrix
    
    def generate_image_histograms(self, magnitudes, orientations):
        '''
        generates histogram over the image with their respective cells.
        
        args:
        `magnitudes`: numpy.ndarray, array with same shape as image having magnitudes of all pixels
        `orientations`: numpy.ndarray, array with same shape as image having angels of all pixels

        returns:
        numpy.ndarray, array of histograms from each cell in the image
        '''
        cell_nr = magnitudes.shape[0] // self.pixels_per_cell[0] # 64//8 = 8
        cell_nc = magnitudes.shape[1] // self.pixels_per_cell[1] # 128//8 = 16

        p_r = self.pixels_per_cell[0]
        p_c = self.pixels_per_cell[1]

        hist_matrices = []

        # First move horizontally then move vertically so that hist_matrices has same dimension of (height, width)
        # Move vertically
        for r in range(cell_nr):

            # Keep track of each histogram that we get vertically
            hists = []

            # Move horizontally
            for c in range(cell_nc):

                # pick 8X8 cells horizontally and vertically for magnitudes and orientations
                # 8X8 cell positions (y1, x1, y2, x2)
                # (0, 0, 7, 7), (8, 0, 15, 7), ..., (j*ppc, i*ppc, (j+1)*ppc-1, (i+1)*ppc-1)
                magnitude_cell = magnitudes[r*p_r:(r+1)*p_r,
                                            c*p_c:(c+1)*p_c]
                orientation_cell = orientations[r*p_r:(r+1)*p_r,
                                                c*p_c:(c+1)*p_c]

                # Generate 9X1 histogram for each 8X8 cell
                histogram = self.generate_histogram(orientation_cell, magnitude_cell)

                # Append it to the existing list of horizontal historgrams
                hists.append(histogram)

            # Append list of horizontal histograms to existing list
            hist_matrices.append(hists)

        return np.array(hist_matrices)
    
    def normalize_histograms(self, histograms):
        '''
        Normalize the hostograms of each cell using a block and normalizing that block by calculating norm and dividing all values in that block by that norm.
        '''
        block_nr = histograms.shape[0] - self.cells_per_block[0] + 1 # 8-2+1=7
        block_nc = histograms.shape[1] - self.cells_per_block[1] + 1 # 16-2+1=15

        c_r = self.cells_per_block[0]
        c_c = self.cells_per_block[1]

        normalized_blocks = []

        # First horizontal and then vertical because dimension should be (height, width) convention
        for r in range(block_nr):

            norm_blocks = []

            for c in range(block_nc):

                # Block positions (y1, x1, y2, x2)
                # (0, 0, 1, 1), (0, 1, 1, 2), ..., (j, i, j+cpb-1, j+cpb-1)
                block = histograms[r:r+c_r, c:c+c_c, ...]
                block = block.reshape((-1, 1)) # (36, 1)
                norm = np.sqrt(np.sum(block**2)) # k = sqrt(a1**2+a2**2+...)
                block = block / norm

                norm_blocks.append(block)

            normalized_blocks.append(norm_blocks)

        return np.array(normalized_blocks)
    
    def hog(self, img_path):
        ''''
        performs the histogram oriented gradient on the image located at `img_path`.

        args:
        `img_path`: Pathlib.Path data, location of the image

        returns:
        numpy.ndarray, features of the image after performing histogram oriented gradients
        '''
        
        # Open the image
        img = self._open_image(img_path)

        # Calculate magnitudes and orientations
        magnitudes, orientations = self.calculate_magnitude_angels(img)

        # Divide the image and calculate histograms for each cell
        histograms = self.generate_image_histograms(magnitudes, orientations)

        # Normalize the histograms with blocks of normization
        norm_histograms = self.normalize_histograms(histograms)

        # Flatten the normalized histograms
        hog_features = norm_histograms.flatten()

        return hog_features

