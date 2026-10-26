"""Wrist RGB object/background relative motion with proprioceptive lift."""
import cv2,numpy as np
from collections import deque
class WristAttachment:
 def __init__(self,rgb,mask,ee):
  self.gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  inner=cv2.erode(mask.astype(np.uint8)*255,np.ones((5,5),np.uint8))
  bg=(1-cv2.dilate(mask.astype(np.uint8),np.ones((31,31),np.uint8)))*255
  bg[:50]=0;bg[380:]=0;bg[:,:140]=0;bg[:,500:]=0
  self.points=cv2.goodFeaturesToTrack(self.gray,80,.01,5,mask=inner)
  self.bg=cv2.goodFeaturesToTrack(self.gray,100,.01,6,mask=bg)
  self.valid=self.points is not None and len(self.points)>=6
  self.initial_count=0 if self.points is None else len(self.points)
  self.initial=None if self.points is None else self.points[:,0].copy()
  self.bg_initial=None if self.bg is None else self.bg[:,0].copy()
  self.parallax_streak=0;self.geometry_window=deque(maxlen=5);self.anchor_locked=False;self.rigid_streak=0;self.static_shift_px=0.;self.origin=np.asarray(ee[:3]);self.confirmed=False;self.streak=0;self.wrist=True;self.motion_window=deque(maxlen=4)
 def begin_lift(self,ee):
  self.origin=np.asarray(ee[:3]);self.streak=0;self.motion_window.clear();self.geometry_window.clear();self.parallax_streak=0
  self.initial=None if self.points is None else self.points[:,0].copy()
  self.bg_initial=None if self.bg is None else self.bg[:,0].copy()
 def lock_attachment(self):
  self.initial=self.points[:,0].copy();self.anchor_locked=True
 def flow(self,gray,points,initial):
  if points is None or not len(points):return None,None,0.
  n,s,e=cv2.calcOpticalFlowPyrLK(self.gray,gray,points,None,winSize=(21,21),maxLevel=3)
  if n is None:return None,None,0.
  back,bs,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,n,None,winSize=(21,21),maxLevel=3)
  if back is None:return None,None,0.
  good=(s[:,0]>0)&(bs[:,0]>0)&(e[:,0]<35)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<1.5)
  return n[good],initial[good],float(np.median(np.linalg.norm(n[good,0]-points[good,0],axis=1))) if np.count_nonzero(good)>=6 else 0.
 def update(self,rgb,ee):
  if not self.valid:return {'valid':False,'reason':'wrist_features_missing'}
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  self.points,self.initial,obj_delta=self.flow(gray,self.points,self.initial)
  self.bg,self.bg_initial,bg_delta=self.flow(gray,self.bg,self.bg_initial)
  self.gray=gray
  if self.points is None or len(self.points)<6:self.valid=False;return {'valid':False,'reason':'wrist_object_lost'}
  obj=float(np.median(np.linalg.norm(self.points[:,0]-self.initial,axis=1)))
  back=float(np.median(np.linalg.norm(self.bg[:,0]-self.bg_initial,axis=1))) if self.bg is not None and len(self.bg)>=6 else 0.
  rise=float(np.asarray(ee[:3])[2]-self.origin[2])
  self.motion_window.append((obj_delta,bg_delta));obj_window=sum(x[0] for x in self.motion_window);bg_window=sum(x[1] for x in self.motion_window)
  self.streak=self.streak+1 if len(self.motion_window)==4 and obj_window<4 and bg_window>6 and rise>.035 else 0
  rigid=rise>.10 and len(self.points)>=12 and len(self.points)>=.35*self.initial_count and obj<25 and obj_window<4 and self.static_shift_px>60
  self.rigid_streak=self.rigid_streak+1 if rigid else 0
  self.geometry_window.append((self.static_shift_px,rise,len(self.points)))
  first=self.geometry_window[0];projection_delta=abs(self.static_shift_px-first[0]);window_rise=rise-first[1]
  parallax=len(self.geometry_window)==5 and len(self.points)>=12 and len(self.points)>=.6*first[2] and obj_window<4 and window_rise>.025 and projection_delta>35
  self.parallax_streak=self.parallax_streak+1 if parallax else 0
  if self.streak>=3 or self.rigid_streak>=3 or self.parallax_streak>=3:self.confirmed=True
  return {'valid':True,'features':len(self.points),'attached_error_px':obj_window,'absolute_object_motion_px':obj,'object_frame_motion_px':obj_delta,'background_frame_motion_px':bg_delta,'background_window_motion_px':bg_window,'background_motion_px':back,'rise_m':rise,'streak':self.streak,'confirmed':self.confirmed,'camera':'wrist','rigid_streak':self.rigid_streak,'static_table_shift_px':self.static_shift_px,'feature_survival':len(self.points)/max(1,self.initial_count),'parallax_streak':self.parallax_streak,'window_rise_m':window_rise,'window_static_shift_px':projection_delta,'anchor_locked':self.anchor_locked,'locked_error_px':obj if self.anchor_locked else None}
