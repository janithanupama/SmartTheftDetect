import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

#%%
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT,IMAGE_WIDTH = 100,100

#%%
def frames_extraction(video_path):
    frame_list = []
    video_reader = cv2.VideoCapture(video_path)

    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT)) #--> Toatl Number of Frames
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)  # skip Frequency

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES,frame_counter*skip_frames_window)

        success,frame = video_reader.read()
        #cv2.imshow("LIVE",frame)
        #cv2.waitKey(1)

        if not success:
            break
        #hi janith
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 21)
        # (T, threshInv) = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
        #hi janith
        # img = cv2.medianBlur(gray,5)
        # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,3)
        #Hey
        resized_frame = cv2.resize(frame,(IMAGE_WIDTH,IMAGE_HEIGHT))
        # cv2.imshow("LIVE",resized_frame)
        # cv2.waitKey(0)
        normalized_frame = resized_frame/255
        frame_list.append(normalized_frame)

    video_reader.release()
    frame_list = np.array(frame_list)
    # cv2.destroyAllWindows()
    return frame_list

# set the path of the image folder
video_folder = r'D:\fyp\FYP VIDEOS EDIT\Thef_Detection\videos'
#video_folder = r'D:\fyp\FYP VIDEOS EDIT\videos'
categories = os.listdir(video_folder)

data = []
label = []
for catagory in categories:
    category_path = os.path.join(video_folder,catagory)
    video_names = os.listdir(category_path)
    for video_name in video_names:
        video_path = os.path.join(category_path,video_name)

        try:
            extract_video = frames_extraction(video_path)
            data.append(extract_video)
            if catagory == "nstealing":
                label.append(0)
            elif catagory == "stealing":
                label.append(1)
        except:
            print(video_path)

data = np.array(data)
label = np.array(label)
label = to_categorical(label)

print(data.shape)
print(label.shape)

np.save('data.npy',data)
np.save('label.npy',label)
print('DATA SAVED !!!')
print('Proceed for Trainning.....')
