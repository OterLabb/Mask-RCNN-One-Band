import json
import os
import datetime
import random
import shutil
from PIL import Image
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

'''
Python script to convert output from ArcGIS Pro's tool 'Export Training Data For Deep Learning'
to Microsoft's popular COCO dataset format often used in deep learning.

Started with code from:
https://github.com/waspinator/pycococreator/blob/master/examples/shapes/shapes_to_coco.py
Huge thanks!

This scripts first creates a new subfolder called 'coco' in the root dir, within this folder,
two new folders are created called 'train' and 'val'. In the training and val folder, images will be copied
and a .json file will be created which can be used with Mask R-CNN. 

1. Set rootDir to the folder where ArcGIS Pro has created training data.
2. Change how many percent of the tiles should be used for validation.
3. Change imageFormat to your format. (May currently only work with .tif)
4. Run script 
'''

# Set base variables
rootDir = r'G:\Arcgis_folder\out_files'
mapFile = 'map.txt'
modelDefinitionFile = 'esri_model_definition.emd'
percentForValidation = 20
imageFormat = '.tif'

# TODO: Perhaps split the dataset in three parts, the third for running evaluation on

# Section to create initial .json file in coco format
data = {
    "president": {
        "name": "Zaphod Beeblebrox",
        "species": "Betelgeusian"
    }
}

INFO = {
        "description": "Example Dataset",
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": 2018,
        "contributor": "waspinator",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
}
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]
CATEGORIES = []

coco_output_val = {
    "info": INFO,
    "licenses": LICENSES,
    "images": [],
    "annotations": [],
    "categories": []
}

coco_output_training = {
    "info": INFO,
    "licenses": LICENSES,
    "images": [],
    "annotations": [],
    "categories": []
}


# Function to create the new folders
def createFolders():
    global trainFolder
    global valFolder
    trainFolder = os.path.join(rootDir, 'coco\\train\\images')
    valFolder = os.path.join(rootDir, 'coco\\val\\images')

    if not os.path.exists(trainFolder):
        os.makedirs(trainFolder)

    if not os.path.exists(valFolder):
        os.makedirs(valFolder)

# Function to copy the images to their respective folders
def copyFilesToTrainAndVal():
    # Copy percentage of images to validation dir
    for item in validationList:
        #print(item) os.path.splitext
        jpgfile = item.replace(os.path.splitext(item), imageFormat)
        shutil.copyfile(os.path.join(rootDir, 'images', jpgfile), os.path.join(valFolder, jpgfile))

    # Copy rest of iamges to train folder
    for item in trainList:
        jpgfile = item.replace(os.path.splitext(item), imageFormat)
        shutil.copyfile(os.path.join(rootDir, 'images', jpgfile), os.path.join(trainFolder, jpgfile))

# Function to create masks for the images
def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        #print(x)
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                #print(pixel_str)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

# Function to create annotations based on the masks
def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=True)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

# Find categories from ArcGIS model definition file
with open(os.path.join(rootDir, modelDefinitionFile)) as f:
    jsonData = (json.loads(f.read()))

categories = [] # Empty list to run a for loop on

classData = jsonData['Classes']
print(classData)

# Make a separate dict for finding class ID later in script
categoriesDict = dict()
def addToCategoryDict(id, classname):
    categoriesDict[id] = classname

# Go through each category and make dictionary's used in this script
for i in classData:
    categories.append(i['ClassName'])

    # Add to categoriesDict
    addToCategoryDict(i['ClassValue'], i['ClassName'])


    newtemplist = {
        'id' : int(i['ClassValue']),
        'name' : i['ClassName']
    }
    coco_output_val["categories"].append(newtemplist)
    coco_output_training["categories"].append(newtemplist)

print(categoriesDict, "categoriesDict")
print(" ")

# Set base variables used in coco output
annotation_id = 1
is_crowd = 0
image_id = 1
annotations = []
image_to_json = []

# Function to create image info which is going to the final .json file
def create_image_info(masked, file, image_id):
    width, height = masked.size

    newfilename = file #file.replace('.tif', '.jpg')

    images = {
        'license': 0,
        'file_name': newfilename,
        'width': width,
        'height': height,
        'id': image_id
    }
    return images

# Create empty list to pass images not used for validation
totalTrainList = list()

# Parse all images used for validation
for i in categories:
    print("Creating annotations for", i)
    subdir = os.path.join(rootDir, 'labels', i)
    print("I made it here")
    files = [x for x in os.listdir(subdir) if x.endswith(".tif")]
    print(len(files))
    print(files)

    createFolders()

    validationList = random.sample(files, int(percentForValidation * len(files) / 100))
    validationLength = len(validationList)

    for file in validationList:
        validationLength = validationLength - 1
        print(validationLength, 'files left in val list')
        mask_images = []
        try:
            mask_images.append(Image.open(os.path.join(subdir, file)).convert('RGB'))
            for masked in mask_images:
                sub_masks = create_sub_masks(masked)
                #print(sub_masks)
                for color, sub_mask in sub_masks.items():
                    #print(file)

                    category_id = list(categoriesDict.keys())[list(categoriesDict.values()).index(i)]
                    annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
                    coco_output_val["annotations"].append(annotation)

                    imageInfo = create_image_info(masked, file, image_id)
                    coco_output_val["images"].append(imageInfo)

                    annotation_id += 1
                    image_id += 1
        except:
            print("Found error image: {0}".format(file))

    # Write out validation.json file
    with open(os.path.join(rootDir, 'coco\\val', 'validation.json'), 'w') as outfile:
        json.dump(coco_output_val, outfile)

    # For items not in validation list, put them in training list
    trainList = [x for x in files if x not in validationList]

    for item in validationList:
        shutil.copyfile(os.path.join(rootDir, 'images', item), os.path.join(valFolder, item))
    for item in trainList:
        shutil.copyfile(os.path.join(rootDir, 'images', item), os.path.join(trainFolder, item))

    for file in trainList:
        totalTrainList.append(os.path.join(subdir,file))
    print(totalTrainList)


# Reset base variables for training files
annotation_id = 1
is_crowd = 0
image_id = 1

trainListLenght = len(totalTrainList)

for file in totalTrainList:
    #print(file)
    trainListLenght = trainListLenght - 1
    print(trainListLenght, 'files left in train list')
    #print(i)
    mask_images = []
    #mask_images.append(Image.open(os.path.join(subdir, file)).convert('RGB'))
    try:
        mask_images.append(Image.open(file).convert('RGB'))
        for masked in mask_images:
            sub_masks = create_sub_masks(masked)
            #print(sub_masks)
            for color, sub_mask in sub_masks.items():

                category_id = list(categoriesDict.keys())[list(categoriesDict.values()).index(i)]
                annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
                coco_output_training["annotations"].append(annotation)

                justFile = file[-13:]

                imageInfo = create_image_info(masked, justFile, image_id)
                coco_output_training["images"].append(imageInfo)

                annotation_id += 1
                image_id += 1
    except:
        print("Found error image: {0}".format(file))

# Write out training.json file
with open(os.path.join(rootDir, 'coco\\train', 'training.json'), 'w') as outfile:
    json.dump(coco_output_training, outfile)

# Print final lenght of val and train files
print(len(trainList), " Train files")
print(len(validationList), ' Validation files')