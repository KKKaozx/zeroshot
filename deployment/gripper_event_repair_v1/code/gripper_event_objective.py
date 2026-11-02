"""One explicit event-aware objective, no new head or decision threshold.

State BCE is retained. Opening/closing surrogate losses use adjacent sigmoid
probabilities and fixed TRAIN-population class balance. No prior command is
inferred from a measured gripper opening; only future-window internal edges
are supervised. This is a candidate objective, not an established repair.
"""
import torch
from torch.nn import functional as F


def fit_event_frequencies(train_commands):
    labels=torch.as_tensor(train_commands)
    if labels.ndim!=2 or labels.shape[1]!=16 or not torch.isin(labels,torch.tensor([-1.,1.],device=labels.device)).all():
        raise ValueError('Original training-only binary command population required')
    opened=labels>0
    events={'opening':(~opened[:,:-1])&opened[:,1:], 'closing':opened[:,:-1]&(~opened[:,1:])}
    rates={key:float(value.double().mean()) for key,value in events.items()}
    if not all(0<value<1 for value in rates.values()):
        raise ValueError('Both actual event directions and non-events required in training')
    return {'fit_partition':'train','windows':len(labels),'edges_per_window':15,'rates':rates}


def gripper_event_loss(logits,commands,frequencies):
    if (logits.ndim!=2 or logits.shape!=commands.shape or logits.shape[1]!=16
            or not torch.isfinite(logits).all() or frequencies.get('fit_partition')!='train'
            or not torch.isin(commands,torch.tensor([-1.,1.],device=commands.device)).all()):
        raise ValueError('Finite H16 logits, original commands and train-only frequencies required')
    opened=commands>0
    state=F.binary_cross_entropy_with_logits(logits,opened.to(logits.dtype))
    before,after=logits[:,:-1],logits[:,1:]
    log_open_before,log_closed_before=F.logsigmoid(before),F.logsigmoid(-before)
    log_open_after,log_closed_after=F.logsigmoid(after),F.logsigmoid(-after)
    #Stable log probabilities for event AND its complement. Avoid log(1-exp(x))
    #cancellation near saturated logits. Non-event alternatives are disjoint.
    probabilities={
        'opening':(log_closed_before+log_open_after,
            torch.logaddexp(log_open_before,log_closed_before+log_closed_after).clamp(max=0),
            (~opened[:,:-1])&opened[:,1:]),
        'closing':(log_open_before+log_closed_after,
            torch.logaddexp(log_closed_before,log_open_before+log_open_after).clamp(max=0),
            opened[:,:-1]&(~opened[:,1:]))}
    losses={}
    for name,(log_event,log_non_event,target) in probabilities.items():
        rate=frequencies['rates'][name]
        if not 0<rate<1: raise ValueError('Invalid frozen training event frequency')
        value=torch.where(target,-log_event*(.5/rate),-log_non_event*(.5/(1-rate)))
        losses[name]=value.mean()
    event=(losses['opening']+losses['closing'])/2
    #One frozen coefficient,1. Both terms have population-scale normalization.
    return state+event, {'state':state,'event':event,**losses}
