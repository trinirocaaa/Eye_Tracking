import cv2
import numpy as np


 
def generate_heatmap(frame, gaze_points): 
    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

    #add gaze points to the heatmap
    for x, y in gaze_points:
    
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            heatmap[y, x] += 3

   
    #normalize heatmap -- for visualization 
    heatmap = cv2.GaussianBlur(heatmap, (99, 99), 0)       
    if np.max(heatmap) > 0:
        heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.uint8)
        
    #color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


    return heatmap


