"""Select five distinct current-image objects by public radial-profile consistency."""
import numpy as np,cv2
from scipy.optimize import least_squares,linear_sum_assignment
from .contours import TABLE
from ..cameras import project

def select_profiles(head,dolls):
 n=len(dolls)
 if n<5 or n>20:return None,{'valid':False,'reason':'candidate_count','count':n}
 g=head.geometry;cost=np.ones((5,n))*100;positions={}
 for rank in range(5):
  for i,d in enumerate(dolls):
   obs=g.descriptor(np.array(d['contour'],np.float32).reshape(-1,1,2));initial=np.array(d['xy'])
   if obs is None:continue
   def descriptor(xy):
    uv=project(g.points[rank]+[xy[0],xy[1],TABLE]).astype(np.float32)
    return g.descriptor(cv2.convexHull(uv))
   def residual(xy):return (descriptor(xy)-obs)*[1,1,.7,.7]
   fit=least_squares(residual,initial,bounds=(initial-.08,initial+.08),diff_step=.001,max_nfev=40);pred=descriptor(fit.x)
   cost[rank,i]=np.linalg.norm((pred[2:]-obs[2:])/obs[2:]);positions[(rank,i)]=fit.x
 _,indices=linear_sum_assignment(cost);best=float(sum(cost[r,i] for r,i in enumerate(indices)));alternatives=[]
 for r,i in enumerate(indices):
  other=cost.copy();other[r,i]=100;rr,cc=linear_sum_assignment(other);alternatives.append(float(other[rr,cc].sum()))
 gap=min(alternatives)-best;maximum=float(max(cost[r,i] for r,i in enumerate(indices)));valid=bool(best/5<.25 and maximum<.4 and gap>.1)
 if not valid:return None,{'valid':False,'mean_residual':best/5,'max_residual':maximum,'assignment_gap':gap,'indices':indices.tolist(),'candidate_count':n,'reason':'profile_assignment_unqualified'}
 ds=[dict(dolls[i],rank=r,calibrated_height_m=[.0501,.0703,.0903,.1103,.1305][r],xy=positions[(r,i)].tolist()) for r,i in enumerate(indices)]
 return (ds if valid else None),{'valid':valid,'mean_residual':best/5,'max_residual':maximum,'assignment_gap':gap,'indices':indices.tolist(),'candidate_count':n,'cost_matrix':cost.tolist()}
