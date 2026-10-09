"""
Plots the STL room geometry over the simplified geometry used by the model and the OpenFOAM case.
"""
import numpy as np, struct, sys, types
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as mp
sys.modules.setdefault('torch', types.ModuleType('torch'))
import os; HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, os.path.dirname(HERE))
from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, DOORS, COLUMNS
def read_stl(fn):
    d=open(fn,'rb').read(); n=struct.unpack('<I',d[80:84])[0]
    a=np.frombuffer(d[84:],dtype=np.dtype([('n','<3f4'),('v','<9f4'),('a','<u2')]),count=n)
    return a['v'].reshape(-1,3,3).astype(float)
def slice_z(v, z0):
    segs=[]
    for t in v:
        d=t[:,2]-z0; pts=[]
        for i,j in ((0,1),(1,2),(2,0)):
            if (d[i]>0)!=(d[j]>0):
                s=d[i]/(d[i]-d[j]); pts.append(t[i]+s*(t[j]-t[i]))
        if len(pts)==2: segs.append((pts[0][:2],pts[1][:2]))
    return segs
def slice_y(v, y0):
    segs=[]
    for t in v:
        d=t[:,1]-y0; pts=[]
        for i,j in ((0,1),(1,2),(2,0)):
            if (d[i]>0)!=(d[j]>0):
                s=d[i]/(d[i]-d[j]); pts.append(t[i]+s*(t[j]-t[i]))
        if len(pts)==2: segs.append((pts[0][[0,2]],pts[1][[0,2]]))
    return segs
room=read_stl("RoomVolume.stl"); walls=read_stl("RoomVolume_Walls.stl"); win=read_stl("Windows.stl"); door=read_stl("Doors.stl")
fig=plt.figure(figsize=(16,15))
ax=fig.add_subplot(3,1,1)
for s in slice_z(walls,1.5): ax.plot(*zip(*s),color="k",lw=0.6)
for s in slice_z(win,1.5): ax.plot(*zip(*s),color="tab:blue",lw=1.0)
for s in slice_z(door,1.0): ax.plot(*zip(*s),color="tab:orange",lw=1.0)
ax.add_patch(mp.Rectangle((ROOM_X[0],ROOM_Y[0]),ROOM_X[1]-ROOM_X[0],ROOM_Y[1]-ROOM_Y[0],fill=False,ec="red",lw=1.2,ls="--"))
for a,b,_,_ in WINDOWS: ax.plot([a,b],[ROOM_Y[1]]*2,color="red",lw=4,alpha=0.5)
for a,b,_,_ in DOORS: ax.plot([a,b],[ROOM_Y[0]]*2,color="red",lw=4,alpha=0.5)
for cx,cy,r,_,_ in COLUMNS: ax.add_patch(mp.Circle((cx,cy),r,fill=False,ec="red",lw=1.2,ls="--"))
ax.set_aspect("equal"); ax.set_xlim(-0.3,15.9); ax.set_ylim(-0.4,9.5)
ax.set_title("Floor plan at z = 1.5 m.  BLACK = STL walls (RoomVolume_Walls.stl), BLUE = Windows.stl, ORANGE = Doors.stl (z=1.0)\n"
             "RED dashed = what the model / OpenFOAM case uses (point_sampler.py)")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
for k,(ylo,yhi,title) in enumerate(((8.4,9.4,"zoom: window wall (y = 8.4 .. 9.4 m)"),(-0.3,1.0,"zoom: door wall (y = -0.3 .. 1.0 m)"))):
    ax=fig.add_subplot(3,1,2+k)
    for s in slice_z(walls,1.5): ax.plot(*zip(*s),color="k",lw=0.7)
    for s in slice_z(win,1.5): ax.plot(*zip(*s),color="tab:blue",lw=1.2)
    for s in slice_z(door,1.0): ax.plot(*zip(*s),color="tab:orange",lw=1.2)
    ax.axhline(ROOM_Y[1] if k==0 else ROOM_Y[0],color="red",ls="--",lw=1)
    for a,b,_,_ in (WINDOWS if k==0 else DOORS): ax.plot([a,b],[(ROOM_Y[1] if k==0 else ROOM_Y[0])]*2,color="red",lw=4,alpha=0.5)
    for cx,cy,r,_,_ in COLUMNS: ax.add_patch(mp.Circle((cx,cy),r,fill=False,ec="red",lw=1.2,ls="--"))
    ax.set_xlim(-0.3,15.9); ax.set_ylim(ylo,yhi); ax.set_title(title); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
fig.tight_layout(); fig.savefig("geometry_overlay.png",dpi=110)
print("ok")
