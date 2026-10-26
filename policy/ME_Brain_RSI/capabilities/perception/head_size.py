"""Resolve ambiguous RGB mask sizes with offline radial profiles and static camera."""
import itertools
import numpy as np,cv2
from scipy.optimize import least_squares
from .wrist_geometry import WristGeometry
from .contours import TABLE
from ..cameras import project

class HeadSize:
 def __init__(self):self.geometry=WristGeometry()
 def assign(self,dolls):
  if len(dolls)!=5:return None,{'valid':False,'reason':'count_not_five'}
  g=self.geometry;cost=np.zeros((5,5))
  for rank in range(5):
   for i,d in enumerate(dolls):
    obs=g.descriptor(np.array(d['contour'],np.float32).reshape(-1,1,2));initial=np.array(d['xy'])
    if obs is None:return None,{'valid':False,'reason':'small_contour'}
    def descriptor(xy):
     uv=project(g.points[rank]+[xy[0],xy[1],TABLE]).astype(np.float32)
     return g.descriptor(cv2.convexHull(uv))
    def residual(xy):return (descriptor(xy)-obs)*[1,1,.7,.7]
    fit=least_squares(residual,initial,bounds=(initial-.08,initial+.08),diff_step=.001,max_nfev=40)
    cost[rank,i]=np.linalg.norm((descriptor(fit.x)[2:]-obs[2:])/obs[2:])
  candidates=sorted((sum(cost[r,cols[r]] for r in range(5)),cols) for cols in itertools.permutations(range(5)))
  score,indices=candidates[0];gap=float(candidates[1][0]-score);maximum=float(max(cost[r,indices[r]] for r in range(5)))
  valid=bool(score/5<.25 and maximum<.4 and gap>.1)
  return ([dolls[i] for i in indices] if valid else None),{'valid':valid,'mean_residual':float(score/5),'max_residual':maximum,'assignment_gap':gap,'indices':list(indices),'cost_matrix':cost.tolist()}

 def locate(self,d,rank,expected_xy=None):
  g=self.geometry;obs=g.descriptor(np.array(d['contour'],np.float32).reshape(-1,1,2));initial=np.array(d['xy'] if expected_xy is None else expected_xy)
  if obs is None:return None,{'valid':False,'reason':'small_contour'}
  def descriptor(xy):
   uv=project(g.points[rank]+[xy[0],xy[1],TABLE]).astype(np.float32)
   return g.descriptor(cv2.convexHull(uv))
  def residual(xy):return (descriptor(xy)-obs)*[1,1,.7,.7]
  fit=least_squares(residual,initial,bounds=(initial-.05,initial+.05),diff_step=.001,max_nfev=40);pred=descriptor(fit.x);center_error=float(np.linalg.norm(pred[:2]-obs[:2]));size_error=float(np.max(np.abs(pred[2:]/obs[2:]-1)));delta=float(np.linalg.norm(fit.x-initial));valid=bool(center_error<5 and size_error<.4 and delta<.05)
  return (fit.x if valid else None),{'valid':valid,'fitted_xy':fit.x.tolist(),'initial_xy':initial.tolist(),'center_error_px':center_error,'size_error':size_error,'shift_m':delta,'initialization':'observed_contour_bottom' if expected_xy is None else 'within_episode_target_xy'}
