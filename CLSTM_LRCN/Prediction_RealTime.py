import cv2
import numpy as np
from create_model import create_model

#%%
SEQUENCE_LENGTH = 50
IMAGE_HEIGHT,IMAGE_WIDTH = 240,320 

#%%
model = create_model(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH)

#%%
# save best weight file 
model.load_weights(r'models\model-005.h5')

#%%
video_reader = cv2.VideoCapture(0)
 
original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames_queue = []
predicted_dict = {0:"Non Stealing",
                  1:"Stealing"}

predicted_class_name =''

frame_count = 0
skip_frequency = 3
while video_reader.isOpened():
    ok, frame = video_reader.read() 
    if not ok:
        break
    
    resized_frame = cv2.resize(frame, (IMAGE_WIDTH,IMAGE_HEIGHT))
    normalized_frame = resized_frame / 255
    
    if frame_count % skip_frequency == 0:
        frames_queue.append(normalized_frame)
        
    if len(frames_queue) == SEQUENCE_LENGTH:
        predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis = 0))[0]
 
        predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = predicted_dict[int(predicted_label)]
        print(predicted_class_name)
        frames_queue = []
    
    frame_count +=1
    cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("VIDEO",frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video_reader.release()
cv2.destroyAllWindows()
