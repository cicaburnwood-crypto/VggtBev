"""Offline evidence visualization; GT is read only for diagnosis, not inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
from PIL import Image

root=Path(__file__).resolve().parent/'repair_evidence'
summary=[]
for scene in ('hm3d_00','procthor_00'):
    audit=json.loads((root/'rgb_replay'/scene/'audit.json').read_text())[1]
    interface=root/'interfaces'/scene
    pose=json.loads((interface/'interfaces/our_model/interface.json').read_text())['predictions'][1]['observation_pose']
    maps=np.load(root/'rgb_replay'/scene/'frame_01.npz')
    extent=float(maps['extent_m'])
    gt=np.load(interface/'geometry.npz');truth=gt['truth'];v=float(gt['voxel_size_m']);low=gt['lower_bound']
    rr,cc=np.nonzero(truth!=255)
    x=low[0]+cc*v;z=low[2]+(len(truth)-1-rr)*v
    dx=np.maximum(np.abs(x-pose['x'])-v/2,0);dz=np.maximum(np.abs(z-pose['z'])-v/2,0)
    nearest=float(np.min(np.hypot(dx,dz)))
    summary.append(dict(scene=scene,gt_body_band_closest_boundary_m=nearest,
                        single=audit['single'],shm=audit['shm']))
    fig,ax=plt.subplots(1,3,figsize=(14,4.7))
    ax[0].imshow(Image.open(interface/'interfaces/our_model/observation_1.jpg'))
    ax[0].set_title('Actual saved RGB after first turn');ax[0].axis('off')
    for a,key,label in ((ax[1],'single','Single prediction'),(ax[2],'shm','SHM fused prediction')):
        a.imshow(maps[key],cmap='gray',vmin=0,vmax=255,extent=(-extent/2,extent/2,-extent/2,extent/2),interpolation='nearest')
        a.add_patch(Circle((0,0),.31213203435596426,fill=False,color='#C7194B',lw=1.5,label='Body circle + 10 cm margin'))
        a.add_patch(Rectangle((-.15,-.15),.3,.3,fill=False,color='#1265DB',lw=1.5,label='Actual 30 x 30 cm body'))
        a.scatter([0],[0],marker='^',s=70,c='#1265DB',edgecolors='white',zorder=5)
        a.set_xlim(-1,1);a.set_ylim(-1,1);a.set_xlabel('Ego right (m)');a.set_ylabel('Ego forward (m)')
        a.set_title(label+' — nearest occupied %.3f m'%audit[key]['closest_occupied_cell_center_m'])
    ax[2].legend(loc='lower left',fontsize=8)
    fig.suptitle(scene+' | center is free; predicted inflation blocks start',fontsize=13)
    fig.tight_layout();fig.savefig(root/f'{scene}_origin.png',dpi=140);plt.close(fig)
(root/'origin_diagnosis.json').write_text(json.dumps(dict(scope='offline evaluator-only GT comparison; not fed to planner',cases=summary),indent=2)+'\n')
print(json.dumps(summary,indent=2))
