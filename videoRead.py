import cv2
import time
#import csv
import numpy as np

# Same command function as streaming, its just now we pass in the file path, nice!
cap = cv2.VideoCapture('video_file/output005.avi')

# FRAMES PER SECOND FOR VIDEO
fps =200		
# Always a good idea to check if the video was acutally there
# If you get an error at thsi step, triple check your file path!!
if cap.isOpened()== False: 
    print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")
    

# While the video is opened
while cap.isOpened():
    
    
    
    # Read the video file.
    ret, frame = cap.read()
    
    # If we got frames, show them.
    if ret == True:
        
        
        

         # Display the frame at same frame rate of recording
        # Watch lecture video for full explanation
        time.sleep(1/fps)
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 960,720)
        cv2.imshow('frame',frame)
 
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break
 
    # Or automatically break this whole loop if the video is over.
    else:
        break
        
cap.release()
# Closes all the frames
cv2.destroyAllWindows()


# npy read
#data = np.load('npy/test.npy',allow_pickle=True)
#data = data.all()

# npz_compressed read
#obj = np.load('x_compressed.npz')
#namelist = obj.zip.namelist()
#obj.zip.extract(namelist[0])
#data = np.load(namelist[0], allow_pickle=True)



#csv read
#with open('csv_file/something.csv') as csv_file:
#    reader = csv.reader(csv_file)
#    mydict = dict(reader)

