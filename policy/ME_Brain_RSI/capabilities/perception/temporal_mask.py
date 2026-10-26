"""Transfer prior semantic identity across one small RGB-observed wrist servo."""
import cv2,numpy as np

def propagate(previous_rgb,previous_mask,current_rgb,expected_center):
 gray0=cv2.cvtColor(previous_rgb,cv2.COLOR_RGB2GRAY);gray1=cv2.cvtColor(current_rgb,cv2.COLOR_RGB2GRAY)
 inner=cv2.erode(previous_mask.astype(np.uint8)*255,np.ones((5,5),np.uint8))
 pts=cv2.goodFeaturesToTrack(gray0,160,.005,4,mask=inner)
 if pts is None or len(pts)<12:return None,{'valid':False,'reason':'prior_features_missing'}
 nxt,ok,err=cv2.calcOpticalFlowPyrLK(gray0,gray1,pts,None,winSize=(31,31),maxLevel=4)
 if nxt is None:return None,{'valid':False,'reason':'forward_flow_missing'}
 back,bok,_=cv2.calcOpticalFlowPyrLK(gray1,gray0,nxt,None,winSize=(31,31),maxLevel=4)
 if back is None:return None,{'valid':False,'reason':'backward_flow_missing'}
 good=(ok[:,0]>0)&(bok[:,0]>0)&(err[:,0]<40)&(np.linalg.norm(back[:,0]-pts[:,0],axis=1)<1.5)
 if good.sum()<12:return None,{'valid':False,'reason':'flow_inliers_missing','tracks':int(good.sum())}
 matrix,inliers=cv2.estimateAffine2D(pts[good,0],nxt[good,0],method=cv2.RANSAC,ransacReprojThreshold=2.5)
 if matrix is None:return None,{'valid':False,'reason':'affine_missing'}
 warped=cv2.warpAffine(previous_mask.astype(np.uint8),matrix,(gray1.shape[1],gray1.shape[0]),flags=cv2.INTER_NEAREST)>0
 yy,xx=np.nonzero(warped);center=np.array([xx.mean(),yy.mean()]) if len(xx) else np.array([-999.,-999.]);ratio=float(warped.sum()/max(1,previous_mask.sum()));distance=float(np.linalg.norm(center-expected_center)) if expected_center is not None else 999.
 scales=np.linalg.svd(matrix[:,:2])[1]
 valid=bool(np.min(scales)>.7 and np.max(scales)<1.4 and np.linalg.det(matrix[:,:2])>0 and inliers.sum()>=12 and inliers.mean()>=.6 and .5<ratio<1.8 and distance<35 and warped.sum()>400)
 diag={'valid':valid,'tracks':int(good.sum()),'inliers':int(inliers.sum()),'inlier_fraction':float(inliers.mean()),'area_ratio':ratio,'geometry_center_error_px':distance,'matrix':matrix.tolist()}
 return (warped if valid else None),diag
