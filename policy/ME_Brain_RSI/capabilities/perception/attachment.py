"""Observed object texture motion against public EE translation and static scene."""
import cv2
import numpy as np
from .contours import TABLE
from ..cameras import project, on_plane

class Attachment:
 def __init__(self,rgb,contour,height,ee):
  self.gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);mask=np.zeros(self.gray.shape,np.uint8)
  cv2.fillPoly(mask,[np.asarray(contour,np.int32)],255)
  mask=cv2.erode(mask,np.ones((3,3),np.uint8))
  self.points=cv2.goodFeaturesToTrack(self.gray,80,.01,4,mask=mask)
  self.valid=self.points is not None and len(self.points)>=6
  self.origin=np.asarray(ee[:3]);self.confirmed=False;self.streak=0;self.evidence={}
  if not self.valid:return
  uv=self.points[:,0]
  self.world=on_plane(uv,TABLE+.5*height)
  self.initial=uv.copy()
 def update(self,rgb,ee):
  if not self.valid:return {'valid':False,'reason':'same_instance_features_missing'}
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  n,s,e=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None,winSize=(21,21),maxLevel=3)
  if n is None:self.valid=False;return {'valid':False}
  back,bs,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,n,None,winSize=(21,21),maxLevel=3)
  if back is None:self.valid=False;return {'valid':False}
  good=(s[:,0]>0)&(bs[:,0]>0)&(e[:,0]<35)&(np.linalg.norm(back[:,0]-self.points[:,0],axis=1)<1.5)
  self.points=n[good];self.world=self.world[good];self.initial=self.initial[good];self.gray=gray
  if len(self.points)<6:self.valid=False;return {'valid':False,'features':len(self.points)}
  delta=np.asarray(ee[:3])-self.origin
  attached=float(np.median(np.linalg.norm(self.points[:,0]-project(self.world+delta),axis=1)))
  static=float(np.median(np.linalg.norm(self.points[:,0]-self.initial,axis=1)))
  observable=attached<8 and static>6 and delta[2]>.035
  self.streak=self.streak+1 if observable else 0
  if self.streak>=3:self.confirmed=True
  self.evidence={'valid':True,'features':len(self.points),'attached_error_px':attached,'static_error_px':static,'rise_m':float(delta[2]),'streak':self.streak,'confirmed':self.confirmed}
  return self.evidence
