"""Render scientific summaries from the independently audited trial artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/panda_spatial_fixed_trial_20261015'
OUT=ROOT/'reports/panda_spatial_fixed_trial_analysis_20261015'


def main():
    analysis=json.loads((OUT/'analysis.json').read_text());assert analysis['verified']
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,field,label,base in [(axes[0],'path_position_cm','Position error (cm)','static_position_cm'),
        (axes[1],'path_rotation_deg','Rotation error (degrees)','static_rotation_deg')]:
        data=analysis['offline'];x=np.arange(2)
        ax.bar(x-.18,[data[s]['metrics'][field] for s in ['train','development']],width=.36,label='Model')
        ax.bar(x+.18,[data[s][base] for s in ['train','development']],width=.36,label='No motion',color='#999999')
        ax.set_xticks(x,['train','development']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.legend()
    counts=analysis['physical']['per_position_successes']
    axes[2].bar(list(counts),list(counts.values()),color=['#278052' if n>=2 else '#c04f3c' for n in counts.values()])
    axes[2].set(ylim=(0,3.5),ylabel='Successful learned rollouts / 3',title='Held-out cube XY positions')
    axes[2].set_yticks(range(4));axes[2].tick_params(axis='x',rotation=25);axes[2].grid(axis='y',alpha=.2)
    fig.savefig(OUT/'trial_summary.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    starts=[r['observation_start'] for r in analysis['by_observation_start'] if r['split']=='train'];x=np.arange(len(starts))
    for offset,split,label in [(-.18,'train','Train'),(.18,'development','Development')]:
        values=[r['position_cm'] for r in analysis['by_observation_start'] if r['split']==split]
        ax.bar(x+offset,values,width=.36,label=label)
    ax.set_xticks(x,[str(t) for t in starts]);ax.set(xlabel='Recorded observation start (expert trajectory step)',
        ylabel='Mean path position error (cm)',title='The initial approach fails even on training observations')
    ax.grid(axis='y',alpha=.2);ax.legend();fig.savefig(OUT/'error_by_observation_start.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,episode in zip(axes.flatten(),counts):
        for seed in [1101,1102,1103]:
            with np.load(RUN/'rollouts'/f'{episode}_seed{seed}.npz') as z:h=z['object_heights']
            # Ground-truth initial cube height is part of the physical rollout receipt.
            report=json.loads((RUN/'rollout_report.json').read_text())
            row=next(r for r in report['results'] if r['episode']==episode and r['sampling_seed']==seed)
            ax.plot(np.arange(1,113)*.1,(h-row['initial_height_m'])*100,label=str(seed))
        ax.axhline(10,color='#777777',linestyle='--',linewidth=1)
        ax.axvspan(10.2,11.2,color='#dddddd',alpha=.4)
        ax.set(title=episode,xlabel='Physical simulation time (s)',ylabel='Cube lift (cm)');ax.grid(alpha=.2);ax.legend(title='Sampling seed')
    fig.savefig(OUT/'learned_physical_traces.png',dpi=170);plt.close(fig)
    # Fixed first sampling draw, all four positions; no successful-run selection.
    from PIL import Image,ImageDraw
    frames=[]
    data=[]
    for episode in counts:
        with np.load(RUN/'rollouts'/f'{episode}_seed1101.npz') as z:data.append(z['images'])
    for t in range(29):
        frame=Image.new('RGB',(448,498),'white');draw=ImageDraw.Draw(frame)
        for i,(episode,images) in enumerate(zip(counts,data)):
            x,y=(i%2)*224,(i//2)*249
            frame.paste(Image.fromarray(images[t]),(x,y));draw.text((x+5,y+229),episode+' learned seed1101',fill='black')
        frames.append(frame)
    frames[0].save(OUT/'all_positions_seed1101.gif',save_all=True,append_images=frames[1:],duration=400,loop=0)
    frames[-1].save(OUT/'all_positions_seed1101_final.png')


if __name__=='__main__':main()
