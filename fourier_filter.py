
import numpy
import os
import csv

from matplotlib import pyplot, patches

from skimage.external import tifffile
from skimage.filters import threshold_mean, threshold_li
from skimage import measure, feature, filters
from PIL import Image

from fnmatch import fnmatch


def fourier_transform(img):
    """Computes the fft of an MxN numpy array

    :params img: The MxN numpy array
    
    :returns : The magnitude and the phase of the fft
    """
    F = numpy.fft.fftshift(numpy.fft.fft2(img))
    return numpy.abs(F), numpy.angle(F), F
    
    
def ellipse_radius(slope, value, array, resolution):
	""" Computes the radius of an ellipse given a slope and both a and b parameters of the ellipse.

	:param slope: Slope of the line.
	:param value: The radius of the ellipse in um/cycle.
	:param array: The array to kone the shape.
	:param resolution: resolution of the image in um.

	returns: The radius of the ellipse and a (x_axis) and b (y_axis) parameters all in pixels.
	"""
	angle = numpy.arctan(slope)
	x_axis = (1 / value) * (array.shape[1] * (1/resolution[0]))
	y_axis = (1 / value) * (array.shape[0] * (1/resolution[1]))
	return (x_axis*y_axis)/numpy.sqrt((x_axis * numpy.sin(angle))**2 + (y_axis * numpy.cos(angle))**2), x_axis, y_axis


def center_image(img):
    """Computes the center of a numpy array
    
    :param img: A numpy array of size MxN
    
    :returns : The center of the array
    """
    if img.shape[0] % 2 == 0 and img.shape[1] % 2 == 0: 
        centerImage = (img.shape[1]/2, img.shape[0]/2)
    elif img.shape[0] % 2 == 0 and img.shape[1] % 2 != 0:
        centerImage = (img.shape[1]/2, (img.shape[0] - 1)/2)
    elif img.shape[0] % 2 != 0 and img.shape[1] % 2 == 0:
        centerImage = ((img.shape[1] - 1)/2, img.shape[0]/2)
    else:
        centerImage = ((img.shape[1] - 1)/2, (img.shape[0] - 1)/2)
    return centerImage
    

def indices(img, center, axis, smaller=True):
    """Returns an MxN array of boolean
    
    :param img : An MxN numpy array
    :param center : The center of the image (x,y)
    :param axis : The semi major axis and the semin minor axis (a,b)
    :param smaller : Boolean
    
    :returns : An MxN numpy array ob boolean
    """
    array = numpy.zeros(img.shape)
    xc, yc = center
    a, b = axis
    if smaller:
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                if ((x-xc)**2 / a**2) + ((y-yc)**2 / b**2) <= 1:
                    array[y,x] = True
                else:
                    array[y,x] = False
                        
    else:
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                if ((x-xc)**2 / a**2) + ((y-yc)**2 / b**2) > 1:
                    array[y,x] = True
                else:
                    array[y,x] = False
    return array
            

def apply_filter(img, indices, f):
    """Apply the filter on every pixel of an MxN array
    
    :param img : A numpy MxN array
    :param indices : A boolean array of size MxN
    :param f : The function to apply to the pixels that meat the boolean condition
    
    :returns : A MxN filtered array
    """
    im = img
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if not indices[y,x]:
                im[y,x] = img[y,x] * f
    return im


def make_binary(img, algorithm):
    """Computes the binary image on an array given an algorith
    
    :param img : A numpy array
    :param algorithm : An algorithm
    
    :returns : The binary array
    """
    algorithms = {"Li" : threshold_li, "mean" : threshold_mean}
    if algorithm not in algorithms:
        print("This method has not yet been implemented.")
    else:
        thres = algorithms[algorithm](img)
        binary = img > thres
        return binary


def gaussian_blur(img, sigma=10, threshold=5e-5, logicalNot=False):
    '''This function performs a gaussian blur on an numpy array then retreives the
    binary mask of the image
    
    :param img : A numpy array
    :param sigma : The standard deviation of the gaussian blur
    :param threshold : The threshold to use for the binary image
    
    :returns : The binary image filtered with gaussian blur
    '''
    gaussianBlur = filters.gaussian(img, sigma=sigma)
    binaryFilter = (gaussianBlur > threshold).astype(int)
    im = gaussianBlur * binaryFilter
    im = im / numpy.amax(im)
    if logicalNot:
        im[im > 0] = 1
        return numpy.logical_not(im)
    else:
        im[im > 0] = 1
        return im
        

def change_values(im, old, new):
    '''This function changes every given value in an numpy array to a new value
    
    :param im : A numpy array
    :param old : The value that needs to be change
    :param new : The value that the old value will take
    
    :returns : An array of the same size
    '''
    im[im == old] = new
    return im


if __name__ == "__main__":
    
    # Try squared sum difference. If the sum is big, the images are not the same.
    
    root = "/Users/Anthony/Downloads/actinImages"
    keepFormat = "*.tif"
    
    fileList, nameList = [],[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, keepFormat):
                fileList.append(os.path.join(path,name))
                nameList.append(name)
    
    currentDir = os.path.split(os.path.realpath(__file__))[0]
    names = ["/07_Glu-Gly.tif", "/01_Block.tif", "/2_Block.tif"]
    img = tifffile.imread(currentDir + names[0])
    im = Image.open(currentDir + names[0])

    for i, file_ in enumerate(fileList):
        # file_ = fileList[0]
        img = tifffile.imread(file_)
        im = Image.open(file_)
        # TEST IMAGES                   GOOD                   ACTINIMAGES
        # channel 0 is -> axons         channel 0 - actin      channel 0 - axons
        # channel 1 is -> actin         channel 1 - axons      channel 1 - dendrites
        # channel 2 is -> dendrite      channel 2 - dendrites  channel 2 - actin
        for chan in range(img.shape[0]):
            img[chan, :, :] = img[chan, :, :] - numpy.amin(img[chan, :, :])

        axons = gaussian_blur(img[0,:,:], logicalNot=True)
        dendrites = gaussian_blur(img[1,:,:])
        actin = gaussian_blur(img[2,:,:])
        mask = (actin*axons*dendrites).astype(numpy.bool_)

        masked = (mask * img[2,:,:])
        masked = masked / numpy.amax(masked)

        FMag, FPhase, F = fourier_transform(img[2,:,:])
        FMagMax = numpy.amax(FMag)
        FMag = numpy.log(FMag)
        
        # FROM HERE
        
        # # Ellipse radius
        # _ , aMax, bMax = ellipse_radius(0, 0.17, img[0,:,:], im.info.get("resolution"))
        # _ , aMin, bMin = ellipse_radius(0, 0.19, img[2,:,:], im.info.get("resolution"))
        #
        # indOut = indices(img[0,:,:], center_image(img[0,:,:]), (aMin, bMin), smaller=False)
        # indIn = indices(img[0,:,:], center_image(img[0,:,:]), (aMax, bMax), smaller=True)
        # ind = indOut * indIn
        #
        # perc = 1.0
        # FMag = apply_filter(F, ind, perc)
        # baseline = numpy.abs(numpy.fft.ifft2(FMag))
        # baseline = baseline / numpy.amax(baseline)
        #
        # perc = 0.3
        # FMag = apply_filter(F, ind, perc)
        # inverseFourier = numpy.abs(numpy.fft.ifft2(FMag))
        # inverseFourier = inverseFourier / numpy.amax(inverseFourier)
        #
        # # masked = change_values(masked, 0, 1)
        # # filtered = change_values(inverseFourier*mask, 0, 1)
        # # dF = (filtered - masked) / masked
        # # mean = numpy.sum(dF) / numpy.sum(mask)
        #
        # SSDBaseline = numpy.sum((inverseFourier - baseline)**2)
        # SSDMask = numpy.sum((inverseFourier - masked)**2)
        #
        # SSIMBaseline = measure.compare_ssim(inverseFourier, baseline)
        # SSIMMask = measure.compare_ssim(inverseFourier, masked)
        
        # TO HERE
        
        # masked = masked.astype(numpy.float32)
        # filtered = (inverseFourier*mask).astype(numpy.float32)
        # tifffile.imsave("/Users/Anthony/Downloads/actinImages/masked/{}".format(nameList[i]),
        #                 numpy.stack([masked, filtered]))
        # pyplot.close()
        # pyplot.show(block=True)

        # sumActin = numpy.sum(actin)
        # sumInverseFourier = numpy.sum(inverseFourier)
        # sumPixels = numpy.sum(dendrites * numpy.logical_not(axons))

        with open(os.path.join(root, "fourierAmplitude.csv"), "a") as csvfile:
                    writer = csv.writer(csvfile, delimiter="\t")
                    writer.writerow([nameList[i], FMagMax])

        print("File : {}/{}".format(i,len(fileList)-1))