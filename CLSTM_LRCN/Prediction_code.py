import cv2
import numpy as np
from create_model import create_model

#%%
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT,IMAGE_WIDTH = 100,100 

#%%
model = create_model(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH)

#%%
# save best weight file 
model.load_weights('final_weights.h5')

#%%

# Prediction Video
video_file_path = 'stealing (212).avi'

video_reader = cv2.VideoCapture(video_file_path)
#video_reader = cv2.VideoCapture(0)
 
original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames_queue = []
predicted_dict = {0:"Non Stealing",
                        1:"Stealing"}
predicted_class_name =''

video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Total Frames : {video_frames_count}')

frame_count = 0
while video_reader.isOpened():
    ok, frame = video_reader.read() 
    if not ok:
        break

    frame_count+=1
    print(frame_count)
    ####
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 21)
    (T, threshInv) = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    ####
    resized_frame = cv2.resize(threshInv, (IMAGE_WIDTH,IMAGE_HEIGHT))
    normalized_frame = resized_frame / 255
 
    frames_queue.append(normalized_frame)
    if len(frames_queue) == SEQUENCE_LENGTH:
        predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis = 0))[0]
 
        predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = predicted_dict[int(predicted_label)]
        print(predicted_class_name)
        frames_queue = []
        
    cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("VIDEO",frame)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    
cv2.destroyAllWindows()
video_reader.release()
