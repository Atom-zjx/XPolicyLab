"""Bounded RGB-only persistence for transient wrist attachment dropouts."""
import numpy as np
class TemporalAttachmentPersistence:
    def __init__(self, max_dropout=5, centroid_tol_m=0.012):
        self.max_dropout=max_dropout; self.centroid_tol_m=centroid_tol_m
        self.prev=None; self.streak=0; self.dropout=0
    def update(self, rank, centroid, wrist_pose, qualified):
        c=np.asarray(centroid,float); p=np.asarray(wrist_pose,float)
        same=self.prev is not None and self.prev['rank']==rank and np.linalg.norm(c-self.prev['centroid'])<=self.centroid_tol_m and np.linalg.norm(p[:3]-self.prev['pose'][:3])<=0.035
        if qualified and (self.prev is None or same): self.streak+=1; self.dropout=0
        elif self.prev is not None and same and self.dropout<self.max_dropout: self.dropout+=1
        else: self.streak=0; self.dropout=0
        if qualified: self.prev={'rank':rank,'centroid':c,'pose':p}
        return self.streak>=2 and self.dropout<=self.max_dropout
