"""One fixed endpoint, paired development scenes; no model-selection loop."""
import numpy as np
from base_verifier import validate_metrics


def compare(candidate,clip,native,protocol):
    inputs = {'dino8':candidate,'clip8_215999':clip,'native212158':native}
    summaries = {name:validate_metrics(dev,protocol) for name,dev in inputs.items()}
    values = {name:np.asarray([r['max_overlap_reward'] for r in dev['episodes']],dtype=np.float64)
              for name,dev in inputs.items()}
    delta = values['dino8']-values['clip8_215999']
    indices = np.random.RandomState(20261023).randint(0,len(delta),(20000,len(delta)))
    interval = np.quantile(delta[indices].mean(1),[.025,.975]).tolist()
    gap = summaries['native212158']['mean_score']-summaries['clip8_215999']['mean_score']
    return dict(scores=summaries,dino_minus_clip_mean=float(delta.mean()),
        paired_scene_bootstrap95=interval,
        dino_better_worse_equal=[int((delta>0).sum()),int((delta<0).sum()),int((delta==0).sum())],
        native_gap_closed_fraction=float(delta.mean()/gap) if gap>0 else None,
        per_scene_gain=[dict(environment_seed=row['environment_seed'],dino_minus_clip=float(d))
                        for row,d in zip(candidate['episodes'],delta)],
        scope='Reused dev20, one training seed; original CLIP onA6000 versus DINO onA40. '
              'Visual backbone and native representation normalization differ. '
              'Scene bootstrap excludes training-seed uncertainty; not final50 or robot grasp success.',
        main_clip_model_replaced=False,final50_accessed=False,automatic_following_action=False,
        interpretation='Report paired effect size and uncertainty; no automatic extra training, variant or seed')
