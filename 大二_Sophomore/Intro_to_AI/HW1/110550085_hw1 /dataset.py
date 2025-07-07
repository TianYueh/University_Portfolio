import os
import cv2

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    # datapath refers to the path where the folder the important files are.
    '''
    First, I create a list named “dataset” to store the images.
    Then for every image in the folders “face” and “nonface”,
    I use cv2 to read it and append it to the dataset, 
    and then label it with 1 or 0.
    '''
    dataset=[]
    path=dataPath+"/face"
    fileptr=os.listdir(path)

    for i in fileptr:
        image=cv2.imread(path+"/"+i, cv2.IMREAD_GRAYSCALE)
        dataset.append((image, 1))
    
    path=dataPath+"/non-face"
    fileptr=os.listdir(path)
    for i in fileptr:
        image=cv2.imread(path+"/"+i, cv2.IMREAD_GRAYSCALE)
        dataset.append((image, 0))


    # raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset
