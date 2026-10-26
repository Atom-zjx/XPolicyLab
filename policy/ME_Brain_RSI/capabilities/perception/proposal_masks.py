"""CPU mask proposals grounded in current RGB semantic boxes, never layout identity."""
import cv2,numpy as np

def iou(a,b):
 a=np.asarray(a);b=np.asarray(b);x=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
 return x/max(1,np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-x)

def proposal_masks(rgb,proposals):
 cv2.setNumThreads(1);merged=[];output=[]
 for p in sorted(proposals,key=lambda p:-p['confidence']):
  if not any(iou(p['box'],q['box'])>.45 for q in merged):merged.append(p)
 for p in merged:
  box=np.asarray(p['box']);x0,y0=np.maximum(0,np.floor(box[:2]-8)).astype(int);x1,y1=np.minimum([640,480],np.ceil(box[2:]+8)).astype(int)
  crop=rgb[y0:y1,x0:x1].copy();h,w=crop.shape[:2]
  if min(h,w)<10:continue
  rect=(max(1,int(box[0]-x0)),max(1,int(box[1]-y0)),min(w-2,int(np.ceil(box[2]-box[0]))),min(h-2,int(np.ceil(box[3]-box[1]))))
  mask=np.zeros((h,w),np.uint8)
  try:cv2.grabCut(crop,mask,rect,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),4,cv2.GC_INIT_WITH_RECT)
  except cv2.error:continue
  fg=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)
  contours,_=cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  if not contours:continue
  contour=max(contours,key=cv2.contourArea)
  if cv2.contourArea(contour)<30:continue
  full=np.zeros((480,640),np.uint8);cv2.fillPoly(full,[contour+[x0,y0]],1)
  yy,xx=np.nonzero(full);refined=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
  output.append((full,{'box':refined,'confidence':p['confidence'],'mask_area':int(full.sum()),'semantic_prompt':'DINO_box_CPU_GrabCut','proposal_box':p['box']}))
 return output
