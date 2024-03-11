"Imports"
import numpy as np 
import imutils
from skimage.transform import rotate ## Image rotation routine
import scipy.fftpack as fft          ## Fast Fourier Transform
import scipy.misc                    ## Contains a package to save numpy arrays as .PNG



## Methods    


"Radon transform method - turns an image into a sinogram (Not used for reconstruction - this"
"is how the original sinogram was generated"
def radon(image, steps):
    theta = np.linspace(0., 180., steps, endpoint=False)
    sinogram = np.zeros((len(theta), image.shape[1]))  # Use image.shape[1] for width
    for i, angle in enumerate(theta):
        rotated = rotate(image, angle, resize=False)  # Set resize=False to prevent resizing

        # Adjust the rotated image to match the original width
        if rotated.shape[1] > image.shape[1]:
            # Clip the rotated image if it's wider than the original
            excess_width = rotated.shape[1] - image.shape[1]
            start = excess_width // 2
            rotated = rotated[:, start:start+image.shape[1]]
        elif rotated.shape[1] < image.shape[1]:
            # Pad the rotated image if it's narrower than the original
            padding = (image.shape[1] - rotated.shape[1]) // 2
            rotated = np.pad(rotated, ((0, 0), (padding, padding)), 'constant')

        # Sum along the vertical axis (axis=0) and assign to the sinogram
        sinogram[i, :] = rotated.sum(axis=0)
    return sinogram, theta


    


"Translate the sinogram to the frequency domain using Fourier Transform"
def fft_translate(projs):
    #Build 1-d FFTs of an array of projections, each projection 1 row of the array.
    return fft.rfft(projs, axis=1)



def fft_translate(projs):
    fft_projs = np.fft.rfft(projs, axis=1)
    return fft_projs

def custom_ramp_filter(ffts, filter_type='ramp'):
    frequencies = np.fft.rfftfreq(ffts.shape[-1])
    if filter_type == 'ramp':
        filter = np.abs(frequencies)
    elif filter_type == 'shepp-logan':
        # Implement Shepp-Logan filter here
        pass
    filtered = ffts * filter[:, None]
    return filtered

def inverse_fft_translate(ffts):
    return np.fft.irfft(ffts, axis=1)

def back_project(sinogram, theta):
    reconstructed = np.zeros((sinogram.shape[1], sinogram.shape[1]))
    radian_angles = np.deg2rad(theta)

    for i, angle in enumerate(radian_angles):
        # Create a 2D array from the 1D sinogram projection
        projection2D = np.tile(sinogram[i], (sinogram.shape[1], 1))
        
        # Rotate the 2D projection back
        rotation = rotate(projection2D, angle, resize=False)

        # Sum the rotated projection to the reconstructed image
        reconstructed += rotation

    return reconstructed



def ramp_filter(ffts):
    #Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5))
    return ffts * ramp

## Statements



"Import the image as a numpy array and display the original sinogram image"
print("Original Sinogram")
sinogram = imutils.imread('request.jpeg')



print("Sinogram")
sinogram, theta = radon(sinogram, steps=1000) # You can adjust the number of steps
imutils.imshow(sinogram)




"Attempt to reconstruct the image directly from the sinogram without any kind of filtering"
print("Reconstruction with no filtering")
unfiltered_reconstruction = back_project(sinogram, theta)




"Use the FFT to translate the sinogram to the Frequency Domain and print the output"
print("Frequency Domain representation of sinogram")
frequency_domain_sinogram = fft_translate(sinogram)




"Filter the frequency domain projections by multiplying each one by the frequency domain ramp filter"
print("Frequency domain projections multipled with a ramp filter")
filtered_frequency_domain_sinogram = ramp_filter(frequency_domain_sinogram)
# imutils.imshow(filtered_frequency_domain_sinogram)
# scipy.misc.imsave('frequencyDomainProjectionsMultipledWithARampFilter.png', 
#                   filtered_frequency_domain_sinogram)



"Use the inverse FFT to return to the spatial domain"
print("Spatial domain representation of ramp filtered sinogram")
filtered_spatial_domain_sinogram = inverse_fft_translate(filtered_frequency_domain_sinogram)
# imutils.imshow(filtered_spatial_domain_sinogram)
# scipy.misc.imsave('spatialDomainRepresentationOfRampFilteredSinogram.png', 
#                   filtered_spatial_domain_sinogram)

"Re-construct the original 2D image by back-projecting the filtered projections"
print("Original, reconstructed image")
reconstructed_image = back_project(unfiltered_reconstruction, theta) # set the first argument to unfiltered reconstruction
imutils.imshow(reconstructed_image)
scipy.misc.imsave('originalReconstructedImage.png', 
                  reconstructed_image)




"Hamming-Windowed Ramp Filter"
print("Hamming-Windowed reconstructed image")
window = np.hamming(566)
hamming = reconstructed_image * window
imutils.imshow(hamming)
scipy.misc.imsave('hammingWindowedReconstructedImage.png', 
                  hamming)

