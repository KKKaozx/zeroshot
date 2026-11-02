"""Record Codex text review of the bounded pilot; this does not create a bank."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'reports/local_paraphrase_pilot_20261025_v2'


def main():
    raw=(PILOT/'candidates.json').read_bytes()
    rows=json.loads(raw)
    # Explicit per-output judgments after reading the original and all five outputs.
    # These are text equivalence judgments by Codex, not human or scene verification.
    judgments=[
        [('accept_text','Same pot, table, back-left corner and placement goal.')]*5,
        [('pending_source','Original "put up" is ambiguous; no video grounding was inspected. Do not choose raise/place/above on its behalf.')]*5,
        [('accept_text','Same arm position relative to bottom-right of red star.'),
         ('accept_text','Same arm position relative to bottom-right of red star.'),
         ('reject','Sentence fragment; missing relation between arm and red star.'),
         ('accept_text','Same arm position relative to bottom-right of red star.'),
         ('accept_text','Same arm position relative to bottom-right of red star.')],
        [('accept_text','Same moved red star and left-of-blue-triangle destination.')]*5,
        [('reject','Changes in-tray containment to on-tray support.'),
         ('accept_text','Inside the tray preserves containment.'),
         ('reject','To the tray does not preserve in-tray containment.'),
         ('reject','Changes in-tray containment to on-tray support.'),
         ('accept_text','Into the tray preserves the placement goal.')],
        [('reject','Changes in-tray containment to on-tray support.'),
         ('accept_text','Same brush-in-tray placement.'),
         ('reject','To the tray weakens in-tray containment.'),
         ('accept_text','Within the tray preserves containment.'),
         ('accept_text','Into the tray preserves the placement goal.')],
        [('reject','Static description drops the requested movement action.'),
         ('accept_text','Close to is an acceptable near relation with same object identifier.'),
         ('accept_text','Same moved sponge and near-rxbar-blueberry placement goal.'),
         ('reject','Static description and adjacent is stronger than unspecified near.'),
         ('accept_text','Same movement to near rxbar blueberry.')],
        [('reject_format','Repeated first/fifth output violates five-distinct-candidate requirement; retain failed group without repairing it.')]*5,
    ]
    reviewed=[]
    for row,decisions in zip(rows,judgments):
        generated=json.loads(row['raw_generation'])['paraphrases']
        assert len(generated)==len(decisions)==5
        items=[dict(candidate=text,text_review_decision=status,evidence=reason,
                    usable_for_training=False) for text,(status,reason) in zip(generated,decisions)]
        reviewed.append(dict(instruction_id=row['instruction_id'],source=row['source'],
            original=row['instruction'],candidates=items,
            complete_five_text_accepted=all(x['text_review_decision']=='accept_text' for x in items)))
    counts={status:sum(c['text_review_decision']==status for r in reviewed for c in r['candidates'])
            for status in ('accept_text','pending_source','reject','reject_format')}
    result=dict(review_type='Codex explicit text-equivalence review; not human or scene validation',
        pilot_candidates_sha256=hashlib.sha256(raw).hexdigest(),counts=counts,
        complete_five_text_accepted_instructions=sum(r['complete_five_text_accepted'] for r in reviewed),
        reviewed_training_bank_created=False,usable_for_training=False,full_generation_admitted=False,
        decision='Do not expand this pilot to the full catalog yet; preserve all failures and resolve quality and ambiguous-source review first.',
        rows=reviewed)
    (PILOT/'text_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for snapshot,source in [('prompt_snapshot.py','code/local_paraphrase_candidates.py'),
                            ('runner_snapshot.py','scripts/pilot_local_paraphrases_20261025.py')]:
        target=PILOT/snapshot
        if target.exists():assert target.read_bytes()==(ROOT/source).read_bytes()
        else:target.write_bytes((ROOT/source).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
