"""Choose a sorted goal row from current RGB object positions, not layout identity."""
import numpy as np

def baseline_row(dolls, radii):
    xy=np.asarray([d['xy'] for d in dolls],float)
    radii=np.asarray(radii,float)
    baseline_relays=sum((d['xy'][0]>=0)!=(i>=2) for i,d in enumerate(dolls) if i!=2)
    candidates=[]
    for start in (-.48,-.36,-.24,-.12,0.):
        ys=np.sort(xy[:,1]); obs_y=float(np.median(ys[1:-1] if len(ys)>=4 else ys)); row_y=float(np.clip(obs_y, -.30, -.14)); row=np.c_[start+.12*np.arange(5),np.full(5,row_y)]
        row[np.abs(row)<1e-9]=0.
        # Center goals can be reached by either arm, as in the prior controller.
        receivers=np.where(row[:,0]<-.04,-1,np.where(row[:,0]>.04,1,np.where(xy[:,0]<0,-1,1)))
        source_arms=np.where(xy[:,0]<0,-1,1)
        relays=int(np.count_nonzero(receivers!=source_arms))
        future_clearance=min(float(np.linalg.norm(row[i]-xy[j])-radii[i]-radii[j]) for i in range(5) for j in range(i+1,5))
        inside=bool(np.max(np.abs(row[:,0])+radii+.015)<.55)
        relay_clearance=min([float(np.linalg.norm(row[i]-[0.,-.30])-radii[i]-radii[j]) for i in range(5) for j in range(i+1,5) if receivers[j]!=source_arms[j]] or [1.])
        valid=inside and future_clearance>.025 and relay_clearance>.025 and relays<=baseline_relays
        distance=float(np.linalg.norm(row-xy,axis=1).sum())
        candidates.append({'start_x':start,'relays':relays,'future_clearance_m':future_clearance,'distance_m':distance,'relay_clearance_m':relay_clearance,'valid':bool(valid),'row':row.tolist()})
    qualified=[c for c in candidates if c['valid']]
    selected=min(qualified,key=lambda c:(c['relays'],c['distance_m']+.1*abs(c['start_x']+.24))) if qualified else dict(candidates[2],row=np.c_[-.24+.12*np.arange(5),np.full(5,-.12)].tolist())
    return np.asarray(selected['row']),{'selected_relays':selected['relays'],'selected_start_x':selected['start_x'],'fallback_original_row':not bool(qualified),'candidates':candidates,'input_source':'current_RGB_ranked_xy_and_offline_CAD_radii'}


from itertools import permutations

def temporal_consensus(history, candidate, tol=.012):
    """Accept a row update only after two nearby RGB estimates."""
    import numpy as np
    c=np.asarray(candidate,float)
    if not history: return c.tolist(), [c.tolist()]
    h=np.asarray(history[-1],float)
    if abs(float(c[:,1].mean()-h[:,1].mean())) <= tol:
        return ((h+c)/2).tolist(), (history+[c.tolist()])[-2:]
    return h.tolist(), (history+[h.tolist()])[-2:]

def choose_row(dolls, radii):
    xy=np.asarray([d['xy'] for d in dolls],float)
    radii=np.asarray(radii,float)
    baseline,prior=baseline_row(dolls,radii)
    source_arms=np.where(xy[:,0]<0,-1,1)
    relay=np.array([0.,-.30])
    plans=[]
    for start in (-.48,-.36,-.24,-.12,0.):
        ys=np.sort(xy[:,1]); obs_y=float(np.median(ys[1:-1] if len(ys)>=4 else ys)); row_y=float(np.clip(obs_y, -.30, -.14)); row=np.c_[start+.12*np.arange(5),np.full(5,row_y)]
        row[np.abs(row)<1e-9]=0.
        if np.max(np.abs(row[:,0])+radii+.015)>=.55:continue
        receivers=np.where(row[:,0]<-.04,-1,np.where(row[:,0]>.04,1,source_arms))
        crosses=receivers!=source_arms
        relays=int(np.count_nonzero(crosses))
        if relays>prior['selected_relays']:continue
        # Dependencies encode actual source occupancy and later relay use.
        edges=set()
        for i in range(5):
            for j in range(5):
                if i==j:continue
                if np.linalg.norm(row[i]-xy[j])-radii[i]-radii[j]<=.025:edges.add((j,i))
                if crosses[j] and np.linalg.norm(row[i]-relay)-radii[i]-radii[j]<=.025:edges.add((j,i))
                if crosses[i] and np.linalg.norm(xy[j]-relay)-radii[i]-radii[j]<=.025:edges.add((j,i))
        for order in permutations(range(5)):
            position={rank:k for k,rank in enumerate(order)}
            if any(position[a]>=position[b] for a,b in edges):continue
            distance=sum(float(np.linalg.norm(xy[i]-relay)+np.linalg.norm(row[i]-relay)) if crosses[i] else float(np.linalg.norm(xy[i]-row[i])) for i in range(5))
            empty=sum(float(np.linalg.norm(row[a]-xy[b])) for a,b in zip(order,order[1:]))
            plans.append({'row':row.tolist(),'order':list(order),'relays':relays,'distance':distance,'empty_distance':empty,'start_x':start,'dependencies':[list(e) for e in sorted(edges)]})
    if not plans:
        return baseline,dict(prior,pick_order=list(range(5)),dependency_plan_fallback=True)
    selected=min(plans,key=lambda p:(p['relays'],p['distance']+.1*abs(p['start_x']+.24),p['empty_distance'],p['order']))
    return np.array(selected['row']),dict(prior,selected_relays=selected['relays'],selected_start_x=selected['start_x'],pick_order=selected['order'],dependency_plan=selected,dependency_plan_fallback=False,fallback_original_row=False,qualified_plans=len(plans))
