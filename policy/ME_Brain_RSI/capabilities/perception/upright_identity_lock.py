"""RGB-only two-observation identity/upright lock; no GT or layout lookup."""
import numpy as np
class UprightIdentityLock:
 def __init__(self,tol_m=.015): self.tol_m=tol_m; self.prev={}; self.streak={}
 def accept(self,rank,d):
  xy=np.asarray(d.get('xy',[0,0]),float); old=self.prev.get(rank)
  same=old is not None and np.linalg.norm(xy-old)<=self.tol_m
  self.streak[rank]=self.streak.get(rank,0)+1 if same else 1
  self.prev[rank]=xy
  return self.streak[rank]>=2
