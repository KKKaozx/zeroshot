"""Training-only eight selected rows; retain the existing training/evaluation padding rules."""
import numpy as np
import torch
from qualification import HERE,read


def prepare(root,bundle):
    from pose_features import batch_features
    from step_training_runner import tokenizer_identity
    data=bundle['data'];api=bundle['api'];model=bundle['model'];tokenizer=bundle['tokenizer']
    selection=read(HERE/'selection.json');rows=selection['rows'];indices=[r['window_index'] for r in rows]
    old=read(root/'oxe_target_pair_short_v1/selection.json')
    assert old['train'][:8]==rows and all(r['source']=='Bridge' and r['role']=='train' for r in rows)
    assert len({r['group_id'] for r in rows})==8
    items=[]
    for row in rows:
        _,start,_,e,_=data.training.store._row(row['window_index'])
        assert all(e[k]==row[k] for k in ['source','group_id','episode_key','payload_sha256']) and int(start)==row['input_row']
        items.append(data.training[row['window_index']])
    texts,images,current,actions,masks=api.collate_batch(items)
    with np.load(HERE/'numeric_reference.npz') as z:
        assert np.array_equal(actions.numpy(),z['targets']) and np.array_equal(masks.numpy(),z['masks'])
        assert np.array_equal(current.numpy(),z['current_features'])
        pose=batch_features(data.training.store,indices,bundle['stats'],'real_pose')
        assert np.allclose(pose,z['real_pose'],rtol=0,atol=1e-6)
    assert tokenizer_identity(tokenizer)==bundle['tokenizer_identity']
    train_tokens=api.tokenise(tokenizer,texts,torch.device('cuda'))
    # Preserve the original 120-window evaluation's padding length, without training/scoring any other row.
    all_text=[]
    for role in ['train','validation']:
        part=data.training if role=='train' else data.validation
        for row in old[role]:all_text.append(part.original_text[row['window_index']])
    eval_tokens=tokenizer(all_text,padding=True,truncation=True,max_length=77,return_tensors='pt')
    eval_tokens={k:v[:8].cuda() for k,v in eval_tokens.items()}
    model.eval()
    with torch.no_grad():
        tv=model.vision_encoder(pixel_values=images.cuda()).last_hidden_state.detach()
        tt=model.text_encoder(**train_tokens).last_hidden_state.detach()
        ev=[];et=[]
        for lo in range(0,8,4):
            ev.append(model.vision_encoder(pixel_values=images[lo:lo+4].cuda()).last_hidden_state.detach())
            et.append(model.text_encoder(**{k:v[lo:lo+4] for k,v in eval_tokens.items()}).last_hidden_state.detach())
    return {'indices':indices,'images':images.cuda(),'train_tokens':train_tokens,'train_visual':tv,'train_language':tt,
        'eval_visual':torch.cat(ev),'eval_language':torch.cat(et),'eval_attention':eval_tokens['attention_mask'],
        'current':current.cuda(),'actions':actions.cuda(),'masks':masks.cuda(),'pose':torch.from_numpy(pose).cuda(),
        'training_token_shape':list(train_tokens['input_ids'].shape),'evaluation_token_shape':list(eval_tokens['input_ids'].shape)}
