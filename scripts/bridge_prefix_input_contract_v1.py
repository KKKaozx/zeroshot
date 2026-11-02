"""Selected-record parser and input checks; no environment/GPU side effects.

Only requested TFRecord records are parsed. No dataset-wide scans or automatic
fallback to a different source/record. A CPU audit verifies this parser against
TensorFlow's official Example parser before use in the GPU preflight.
"""
import hashlib
import io
from pathlib import Path
import struct
import numpy as np
from PIL import Image


GRID = np.linspace(0,223,8,dtype=np.int64)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(8*1024**2),b""):h.update(part)
    return h.hexdigest()


def varint(stream,end):
    value=0
    for shift in range(0,70,7):
        if stream.tell()>=end:raise ValueError("Truncated protobuf varint")
        b=stream.read(1)[0];value|=(b&127)<<shift
        if b<128:return value
    raise ValueError("Overlong protobuf varint")


def fields(stream,end):
    while stream.tell()<end:
        tag=varint(stream,end);number,wire=tag>>3,tag&7
        if number<=0:raise ValueError("Invalid protobuf field")
        if wire==2:
            n=varint(stream,end);a=stream.tell();b=a+n
            if b>end:raise ValueError("Truncated protobuf payload")
            yield number,a,b
            stream.seek(b)
        elif wire==0:varint(stream,end)
        elif wire in (1,5):stream.seek(8 if wire==1 else 4,1)
        else:raise ValueError("Unsupported protobuf wire type")
    if stream.tell()!=end:raise ValueError("Protobuf boundary mismatch")


def feature(raw):
    stream=io.BytesIO(raw);result=None
    for kind,a,b in fields(stream,len(raw)):
        if result is not None or kind not in (1,2,3):raise ValueError("Unexpected Feature kind")
        stream.seek(a);values=[]
        for number,c,d in fields(stream,b):
            if number!=1:raise ValueError("Unexpected feature list field")
            stream.seek(c)
            if kind==1:values.append(stream.read(d-c))
            elif kind==2:
                if (d-c)%4:raise ValueError("Invalid packed floats")
                values.extend(np.frombuffer(stream.read(d-c),dtype="<f4").tolist())
            else:
                while stream.tell()<d:
                    n=varint(stream,d);values.append(n if n<2**63 else n-2**64)
        result=values
    if result is None:raise ValueError("Empty Feature")
    return result


def parse_record(payload):
    required={"steps/observation/state","steps/action","steps/observation/image_0",
        "steps/language_instruction","steps/is_first","steps/is_last",
        "episode_metadata/file_path","episode_metadata/episode_id"}
    stream=io.BytesIO(payload);out={}
    for outer,a,b in fields(stream,len(payload)):
        if outer!=1:continue
        stream.seek(a)
        for entry,c,d in fields(stream,b):
            if entry!=1:continue
            stream.seek(c);name=span=None
            for number,e,f in fields(stream,d):
                if number==1:stream.seek(e);name=stream.read(f-e).decode()
                elif number==2:span=e,f
            if name in required:
                if name in out or span is None:raise ValueError("Duplicate/missing feature")
                stream.seek(span[0]);out[name]=feature(stream.read(span[1]-span[0]))
            stream.seek(d)
        stream.seek(b)
    if set(out)!=required:raise ValueError("Required Bridge schema absent")
    return out


def records_at(path,indices):
    wanted=set(map(int,indices))
    if not wanted or min(wanted)<0:raise ValueError("Explicit record indices required")
    size=Path(path).stat().st_size
    with Path(path).open("rb") as stream:
        for i in range(max(wanted)+1):
            header=stream.read(12)
            if len(header)!=12:raise ValueError("TFRecord index unavailable")
            length=struct.unpack('<Q',header[:8])[0];start=stream.tell()
            if start+length+4>size:raise ValueError("Truncated TFRecord")
            if i in wanted:
                if not length:raise ValueError("Requested record is an empty placeholder")
                payload=stream.read(length)
                yield i,payload,parse_record(payload)
            stream.seek(start+length+4)


def episode_arrays(values,expected,reverse_scan):
    n=expected['steps']
    states=np.asarray(values['steps/observation/state'],dtype='<f4').reshape(-1,7)
    actions=np.asarray(values['steps/action'],dtype='<f4').reshape(-1,7)
    if states.shape!=actions.shape or states.shape!=(n,7):raise ValueError("Episode length changed")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():raise ValueError("Nonfinite episode")
    if len(values['steps/observation/image_0'])!=n or len(values['steps/language_instruction'])!=n:
        raise ValueError("Observation field lengths differ")
    if values['steps/is_first']!=[1]+[0]*(n-1) or values['steps/is_last']!=[0]*(n-1)+[1]:
        raise ValueError("Invalid first/last markers")
    if values['episode_metadata/file_path']!=[expected['origin_file_path'].encode()] or values['episode_metadata/episode_id']!=[expected['episode_id']]:
        raise ValueError("Source episode identity changed")
    if any(' '.join(s.decode().lower().split())!=expected['instruction'] for s in values['steps/language_instruction']):
        raise ValueError("Instruction identity changed")
    grip=reverse_scan(actions[:-1,6])
    if not np.isin(grip,(0.,1.)).all():raise ValueError("Unresolved valid gripper tail; no silent threshold/replacement")
    return states,actions,grip


def image_contract(values,start,prepare_image):
    encoded=values['steps/observation/image_0'][start]
    rgb=np.asarray(Image.open(io.BytesIO(encoded)).convert('RGB'))
    if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.std()<=1.:raise ValueError("Invalid/blank RGB")
    image=prepare_image(rgb)
    if tuple(image.shape)!=(3,224,224) or not np.isfinite(image.numpy()).all():
        raise ValueError("Invalid normalized image")
    meta=dict(encoded_image_sha256=hashlib.sha256(encoded).hexdigest(),
        decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),rgb_shape=list(rgb.shape),
        instruction_sha256=hashlib.sha256(values['steps/language_instruction'][start]).hexdigest(),
        instruction=values['steps/language_instruction'][start].decode())
    grid=image.numpy()[:,GRID][:,:,GRID].copy()
    return image,meta,grid
