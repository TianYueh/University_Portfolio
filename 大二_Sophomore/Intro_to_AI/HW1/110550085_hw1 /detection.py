import os
import cv2
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    '''
    First, I open the detect file to read the text and
    how many rectangles I should draw on the picture.
    Then, I use cv2 to read the picture by colored and 
    grayscale, and for every rectangle, I cut that part
    and turn that into a 19*19 square, then use the
    classify function to test it with the classifier I
    create. Finally, I use cv2 to draw a 1px wide 
    rectangle on the border of the given range in the
    .txt file.
    '''
    uberPath='./'
    Path=uberPath+dataPath
    detectPath=uberPath+"data/detect/"

    fileptr=open(Path, "r")
    while(True):
        tmp=fileptr.readline()
        if(not tmp):
           break
        tmp=tmp.split( )
        # tmp[1] is the number of rectangles to draw.
        num=int(tmp[1])
        # tmp[0] is the name of the jpg file.
        image=cv2.imread(detectPath+tmp[0])
        grayimage=cv2.imread(detectPath+tmp[0], cv2.IMREAD_GRAYSCALE)
        for i in range(num):
            arr=fileptr.readline().split( )
            x=int(arr[0])
            y=int(arr[1])
            width=int(arr[2])
            height=int(arr[3])
            obj=grayimage[y:y+height, x:x+width]
            obj=cv2.resize(obj, (19, 19))
            draw=clf.classify(obj)

            if (draw==True):
                cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 1)
            else:
                cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 1)
            if(i==num-1):
              cv2.imshow("Result", image)
              # waitKey to make sure that every images can be seen.
              cv2.waitKey(0)
    fileptr.close()
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)
