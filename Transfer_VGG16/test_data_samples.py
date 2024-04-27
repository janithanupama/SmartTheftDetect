import numpy as np
import cv2

data = np.load('data.npy')
target = np.load('label.npy')
print("Data Loaded")
#%%
print(data.shape)
print(target.shape)

#%%
data_sample = data[10,:,:,:]

n_frames = data_sample.shape[0]

for i in range(n_frames):
    frame = data_sample[i,:,:]
    frame = frame*255
    cv2.imshow("LIve",frame.astype(np.uint8))
    key = cv2.waitKey(5)
    if key==ord('q'):
        break
    
cv2.destroyAllWindows()
    
    
