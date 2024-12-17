import imageio
from skimage.measure import label, regionprops
from skimage import io, color, filters, morphology, measure
from skimage.transform import rescale
from skimage.filters import threshold_otsu
from skimage.filters import frangi
from collections import Counter
import os
import numpy as np
import pandas as pd
from skimage.morphology import remove_small_objects, binary_closing,skeletonize
import pydicom

def threshold_segmentation(image, block_size,threshold_value):
    # Convert the image to grayscale if it's not
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Apply thresholding
    #binary_image = image > threshold_value

    # Example with Otsu's method
    binary_image3 = image > filters.threshold_otsu(image)

    # Example with adaptive thresholding
    binary_image1 =  filters.threshold_local(image, block_size)
    binary_image1= binary_image1 > threshold_value
    binary_image2 = image> threshold_value

    return binary_image1, binary_image2, binary_image3 

def windowing_based_HU(dicom_slice, window_min = 130, window_max = 600,block_size=1, threshold_value=.125,min_size = 20,border_pixels = 10, border_pixels_2=20):

    # Define window levels
    window_min = window_min
    window_max = window_max
    
    # Clip pixel values to the desired window range
    windowed_image = np.clip(dicom_slice, window_min, window_max)

    # Normalize pixel values to range [0, 1]
    windowed_image = (windowed_image - window_min) / (window_max - window_min)

    #thresholding
    binary_result1, binary_result2, binary_result3  = threshold_segmentation(windowed_image, block_size=block_size, threshold_value=threshold_value)

    # Assuming binary_image is your thresholded binary image
    binary_image_cleaned = remove_small_objects(binary_result1, min_size=30)  # Adjust min_size as needed
    ##NOT USING
    skeleton = skeletonize(binary_image_cleaned )
    # Assuming skeleton is your skeletonized binary image
    labeled_image = measure.label(skeleton)
    # Get region properties for each labeled region
    regions = measure.regionprops(labeled_image)
    # Find the region with the largest length
    largest_length = 0
    largest_length_region = None

    for region in regions:
        if region.major_axis_length > largest_length:
            largest_length = region.major_axis_length
            largest_length_region = region

    # Extract the coordinates of the largest length region
    rr, cc = draw.polygon(largest_length_region.coords[:, 0], largest_length_region.coords[:, 1])

    # Create an image with the same shape as binary_image_cleaned and draw the region on it
    mapped_image_cleaned = np.zeros_like(binary_image_cleaned)
    mapped_image_cleaned[rr, cc] = binary_image_cleaned[rr, cc]

    min_size = min_size  # Adjust this threshold as needed

    # Remove small connected components
    filtered_image = remove_small_objects(binary_image_cleaned > 0, min_size=min_size)



    normalized_image = (dicom_slice - np.min(dicom_slice)) / (np.max(dicom_slice) - np.min(dicom_slice))

    indices = np.argwhere(filtered_image == 1)

    border_pixels = border_pixels 

    # Get the shape of the original array
    rows, cols = filtered_image.shape

    # Create a new array with the same shape
    bordered_arr = np.zeros((rows, cols), dtype=filtered_image.dtype)

    # Add the existing 1s to the new array with the border
    for index in indices:
        row, col = index
        bordered_arr[row:row + border_pixels, col:col + border_pixels] = 1
        bordered_arr[row - border_pixels:row, col - border_pixels:col] = 1

    labeled_arr, num_labels = label(bordered_arr, connectivity=1, return_num=True)

    # Calculate properties of each labeled region
    regions = regionprops(labeled_arr)

    # Sort regions by their areas in descending order
    sorted_regions = sorted(regions, key=lambda region: region.area, reverse=True)

    # Get the two largest regions
    largest_regions = sorted_regions[:1]

    # Create a new binary image containing only the largest regions
    largest_regions_image = np.zeros_like(bordered_arr)

    for region in largest_regions:
        for coords in region.coords:
            largest_regions_image[coords[0], coords[1]] = 1    
    
    artery = normalized_image*largest_regions_image    
    artery[0,0] = 1
    return artery,mapped_image_cleaned, dicom_slice

def folder_contains_artery_mpr_or_curved(file_name, file_path, artery_name):
    # Check if the file starts with "SC"
    if file_name.startswith("SC"):
        # Read the DICOM file
        dicom_info = pydicom.dcmread(file_path)

        # Check if the artery name and "MPR" or "Curved" are in Series Description
        if hasattr(dicom_info, 'SeriesDescription') and dicom_info.SeriesDescription and artery_name in dicom_info.SeriesDescription and ("MPR" in dicom_info.SeriesDescription or "Curved" in dicom_info.SeriesDescription):
            # Return True if artery MPR or Curved DICOM files are found
            return True

    # Return False if no artery MPR or Curved DICOM files are found
    return False

def fetch_artery_with_bounday_all_slices_3ch(patient_ids, dicom_ids,dicom_fold_num, destination_dir, file_extension='_rgb.png', window_min = 130, window_max = 600,block_size=1, threshold_value=.125,min_size = 20,border_pixels = 1000000, border_pixels_2 = 20):
    for patient_id, dicom_id,dicom_fold_num in zip(patient_ids, dicom_ids,dicom_fold_num):
        source_path = os.path.join('E:/Dicom', patient_id, dicom_id)
        if os.path.exists(source_path):
            destination_path = os.path.join(destination_dir, f"{patient_id}_{dicom_fold_num}")
            os.makedirs(destination_path, exist_ok=True)
            for folder_name, subfolders, files in os.walk(source_path):
                for file_name in files:
                    file_path = os.path.join(folder_name, file_name)
                    # Check if the folder contains MPR or Curved files
                    if folder_contains_artery_mpr_or_curved(file_name, file_path):
                        dicom_slice = imageio.imread(file_path)
                        dicom_slice = np.clip(dicom_slice, -1000, 2000)
                        artery,mapped_image_cleaned, dicom_slice=windowing_based_HU(dicom_slice, window_min = window_min, window_max = window_max,block_size=block_size, threshold_value=threshold_value,min_size =min_size,border_pixels = border_pixels, border_pixels_2=border_pixels_2)
                        output_file = os.path.join(destination_path, os.path.basename(file_path) + file_extension)
                        # Scale pixel values to the range [0, 255]
                        artery_rgb = np.stack((artery,) * 3, axis=-1)
                        artery_rgb = (artery_rgb - artery_rgb.min()) / (artery_rgb.max() - artery_rgb.min()) * 255
                        artery_rgb = artery_rgb.astype('uint8')
                        output_file = os.path.join(destination_path, os.path.basename(file_path) + file_extension)
                        # Save the image
                        imageio.imwrite(output_file, artery_rgb)

    return  artery,mapped_image_cleaned, dicom_slice 



artery_to_process = 'LAD'
folder_name=''  ## where you saved patint id and dicom id
file_name =''           ##final labels
file_path = os.path.join(folder_name, file_name)
result = pd.read_excel(file_path)

class_1_df = result[result['max_severity'] == 1]
class_2_df = result[result['max_severity'] == 2]


class_1_patient_ids = class_1_df['Patient_ID'].tolist()
class_2_patient_ids = class_2_df['Patient_ID'].tolist()

print(len(class_1_patient_ids))
print(len(class_2_patient_ids))

class_1_dicom_ids = class_1_df['Dicom_id'].tolist()
class_2_dicom_ids = class_2_df['Dicom_id'].tolist()



class_1_dicom_fold_num = class_1_df['Dicom_folder_num'].tolist()
class_2_dicom_fold_num = class_2_df['Dicom_folder_num'].tolist()


process_dir = ''###where slices are saved
print(process_dir)

# Create directories for class 1 and class 2 patients
class_1_destination_dir = os.path.join(process_dir, 'Class_1')
class_2_destination_dir = os.path.join(process_dir, 'Class_2')

# Create the directories if they don't exist
os.makedirs(class_1_destination_dir, exist_ok=True)
os.makedirs(class_2_destination_dir, exist_ok=True)

nn=50
artery=fetch_artery_with_bounday_all_slices_3ch(class_1_patient_ids[:], class_1_dicom_ids[:], class_1_dicom_fold_num[:],class_1_destination_dir,artery_to_process,window_min =-100, window_max = 600,block_size=1, threshold_value=.0001,min_size = 10,border_pixels = 10, border_pixels_2=20)
artery=fetch_artery_with_bounday_all_slices_3ch(class_2_patient_ids[:], class_2_dicom_ids[:], class_2_dicom_fold_num[:],class_2_destination_dir,artery_to_process,window_min =-100, window_max = 600,block_size=1, threshold_value=.0001,min_size = 10,border_pixels = 10, border_pixels_2=20)