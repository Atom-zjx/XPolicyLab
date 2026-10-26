"""CPU RGB contour extraction. No simulator or model dependency."""
import cv2
import numpy as np
cv2.setNumThreads(1)
from ..cameras import head_camera, ray, on_plane
TABLE=.765

def refine_mask(mask,proposal):
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return None
    contour=max(contours,key=cv2.contourArea)
    if cv2.contourArea(contour)<25:return None
    points=contour[:,0,:]
    top_v,bottom_v=points[:,1].min(),points[:,1].max()
    top_u=float(np.median(points[points[:,1]<=top_v+2,0]))
    bottom_u=float(np.median(points[points[:,1]>=bottom_v-2,0]))
    CAM=head_camera().center
    bottom=on_plane([[bottom_u,bottom_v]],TABLE)[0]
    t=ray(top_u,top_v)
    scale=np.dot(bottom[:2]-CAM[:2],t[:2])/np.dot(t[:2],t[:2])
    height=float((CAM+scale*t)[2]-TABLE)
    return {**proposal,'contour':points.tolist(),'top_pixel':[top_u,int(top_v)],'bottom_pixel':[bottom_u,int(bottom_v)],'xy':bottom[:2].tolist(),'height_m':height,'height_source':'rgb_sam3_contour_vertical_fit','geometric_valid':bool(-.55<bottom[0]<.55 and -.40<bottom[1]<.12 and .02<height<.30)}
