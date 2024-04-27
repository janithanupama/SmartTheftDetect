import tensorflow as tf
import cv2
import numpy as np

'''
Create interpreter, allocate tensors
'''
tflite_interpreter = tf.lite.Interpreter(model_path="model_tflite.tflite")
tflite_interpreter.allocate_tensors()

'''
Check input/output details
'''
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

'''
Run prediction (optional), input_array has input's shape and dtype
'''
# tflite_interpreter.set_tensor(input_details[0]['index'], input_array)
# tflite_interpreter.invoke()
# output_array = tflite_interpreter.get_tensor(output_details[0]['index'])

'''
This gives a list of dictionaries. 
'''
# tensor_details = tflite_interpreter.get_tensor_details()

# for dict in tensor_details:
#     i = dict['index']
#     tensor_name = dict['name']
#     scales = dict['quantization_parameters']['scales']
#     zero_points = dict['quantization_parameters']['zero_points']
#     tensor = tflite_interpreter.tensor(i)()

#     print(i, type, name, scales.shape, zero_points.shape, tensor.shape)

#     '''
#     See note below
#   '''
    
#%%
import cv2
import numpy as np
# from create_model import create_model
# import tensorflow as tf

#%%
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT,IMAGE_WIDTH = 224,224 

#%%
#model = create_model(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH)

#%%
video_reader = cv2.VideoCapture('1.avi')
 
original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames_queue = []
predicted_dict = {0:"nstealing",
                  1:"stealing"}

predicted_class_name =''

frame_count = 0
skip_frequency = 3
while video_reader.isOpened():
    ok, frame = video_reader.read() 
    if not ok:
        break
    img_show = frame.copy()
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(frame, (IMAGE_WIDTH,IMAGE_HEIGHT))
    normalized_frame = resized_frame / 255
    
    if frame_count % skip_frequency == 0:
        frames_queue.append(normalized_frame)
        
    if len(frames_queue) == SEQUENCE_LENGTH:
        input_array = np.expand_dims(frames_queue, axis = 0)
        input_array = np.array(input_array, dtype='float32')
        input_array = input_array.reshape(1,20,224,224,1)
#         input_array = input_array.reshape(input_array.shape[:,:,:],1)
        print(input_array.shape)
        tflite_interpreter.set_tensor(input_details[0]['index'], input_array)
        tflite_interpreter.invoke()
        output_array = tflite_interpreter.get_tensor(output_details[0]['index'])
        predicted_labels_probabilities = output_array
        
        #predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis = 0))[0]
 
        predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = predicted_dict[int(predicted_label)]
        print(predicted_class_name)
        frames_queue = []
    
    frame_count +=1
    cv2.putText(img_show, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("VIDEO",img_show)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video_reader.release()
cv2.destroyAllWindows()