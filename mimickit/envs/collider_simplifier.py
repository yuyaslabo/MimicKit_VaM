"""Offline, conservative trial simplification for VaM compound-box MJCF assets.

Examples:
  python collider_simplifier.py --input data/assets/objects/vam_env_boxes
  python collider_simplifier.py --input data/assets/objects/vam_env_boxes --output data/assets/objects/vam_env_boxes_preview

Does not overwrite originals. Used by vam_environment_boxes_simplify_test.py.
"""
import argparse
import copy
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _corners(center, half, mat):
    signs = np.array([(a,b,c) for a in (-1.,1.) for b in (-1.,1.) for c in (-1.,1.)])
    return center + (signs*half) @ mat.T


def _box(e):
    try:
        center = np.fromstring(e.attrib.get('pos','0 0 0'), sep=' ', dtype=np.float64)
        half = np.fromstring(e.attrib['size'], sep=' ', dtype=np.float64)
        quat = np.fromstring(e.attrib.get('quat','1 0 0 0'), sep=' ', dtype=np.float64)
        if len(center)!=3 or len(half)!=3 or len(quat)!=4 or min(half)<=0 or not np.isfinite(np.r_[center, half, quat]).all():
            return None
        quat = quat/np.linalg.norm(quat)
        mat = Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]]).as_matrix()
        return {'center':center,'half':half,'mat':mat,'geom':copy.deepcopy(e),'ids':[e.get('name','unnamed')]}
    except (KeyError,ValueError,ZeroDivisionError):
        return None


def _bounds_in_frame(b, origin, frame):
    vertices = _corners(b['center'],b['half'],b['mat'])
    coords = (vertices-origin) @ frame
    return coords.min(axis=0), coords.max(axis=0)


def _merge_pair(a,b,gap,delta,rot_deg,volume_ratio,overlap_ratio):
    relative = a['mat'].T @ b['mat']
    # Direct axes only: do not merge 90-degree rotated boxes, even when cuboids overlap.
    cos_min = math.cos(math.radians(rot_deg))
    if np.min(np.diag(relative)) < cos_min:
        return None
    center=a['center']; frame=a['mat']
    al,ah=_bounds_in_frame(a,center,frame)
    bl,bh=_bounds_in_frame(b,center,frame)
    lo=np.minimum(al,bl); hi=np.maximum(ah,bh)
    lengths_a=ah-al; lengths_b=bh-bl
    overlap=np.maximum(0.,np.minimum(ah,bh)-np.maximum(al,bl))
    axes_touch=[]
    for axis in range(3):
        sep=max(bl[axis]-ah[axis],al[axis]-bh[axis],0.)
        if sep>gap:
            return None
        if sep>1e-8 or overlap[axis]/min(lengths_a[axis],lengths_b[axis])<overlap_ratio:
            axes_touch.append(axis)
    # All three overlapping: near-duplicate / overlapping boxes.
    # Otherwise at most one axis is allowed to differ substantially, avoiding L-shaped unions.
    if len(axes_touch)>1:
        return None
    candidate_axes = axes_touch if axes_touch else [int(np.argmin(overlap/np.minimum(lengths_a,lengths_b)))]
    along=candidate_axes[0]
    for axis in range(3):
        if axis == along:
            continue
        if abs(lengths_a[axis]-lengths_b[axis]) > delta:
            return None
        if overlap[axis]/min(lengths_a[axis],lengths_b[axis]) < overlap_ratio:
            return None
    vol_a=float(np.prod(lengths_a)); vol_b=float(np.prod(lengths_b))
    inter=float(np.prod(overlap)); occupied=vol_a+vol_b-inter
    cover=float(np.prod(hi-lo))
    if occupied<=1e-10 or cover/occupied > volume_ratio:
        return None
    merged=copy.deepcopy(a)
    merged['center']= center+frame @ ((hi+lo)*0.5)
    merged['half']=(hi-lo)*0.5
    merged['ids']=a['ids']+b['ids']
    return merged


def simplify_mjcf(input_path,output_path,thin=0.08,gap=0.03,size_tolerance=0.10,
                  rotation_tolerance=3.0,max_volume_ratio=1.18,overlap_ratio=0.70):
    """Return dict with count/reasons; originals are never modified."""
    tree=ET.parse(input_path); root=tree.getroot()
    boxes=[]; other=0
    for body in root.findall('.//body'):
        # Existing assets contain a single root body, but keep unrelated/unsupported geoms.
        for geom in list(body.findall('geom')):
            if geom.get('type')!='box':
                other+=1; continue
            b=_box(geom)
            if b is None:
                other+=1; continue
            b['body']=body
            boxes.append(b)
    original=len(boxes)
    dropped=[]; remaining=[]
    for b in boxes:
        dims=np.sort(2*b['half'])
        if dims[0] < thin and dims[1] < thin:
            dropped.append(b)
        else:
            remaining.append(b)
    # Only merge within same rigid body; no other geom kinds are modified.
    groups={}
    for b in remaining:
        groups.setdefault(id(b['body']),[]).append(b)
    merged=[]; merge_events=[]
    for group in groups.values():
        while True:
            found=False
            for i in range(len(group)):
                for j in range(i+1,len(group)):
                    merged_box=_merge_pair(group[i],group[j],gap,size_tolerance,rotation_tolerance,
                                            max_volume_ratio,overlap_ratio)
                    if merged_box is not None:
                        merged_box['body']=group[i]['body']
                        merge_events.append((tuple(group[i]['ids']),tuple(group[j]['ids'])))
                        group=[b for k,b in enumerate(group) if k not in (i,j)]+[merged_box]
                        found=True; break
                if found: break
            if not found: break
        merged.extend(group)
    # Replace only supported box geoms; preserve XML hierarchy and other fields.
    for b in boxes:
        b['body'].remove(next(g for g in b['body'].findall('geom') if g is not None and g.get('name') == b['geom'].get('name')))
    total_original_mass=sum(float(b['geom'].get('mass','0')) for b in boxes)
    # If there were no explicit masses, preserve original attributes (XML density/default semantics).
    for index,b in enumerate(merged):
        e=b['geom']
        e.set('name',f'simplified_{index:03d}')
        e.set('pos',' '.join(f'{v:.7f}' for v in b['center']))
        e.set('size',' '.join(f'{v:.7f}' for v in b['half']))
        q=Rotation.from_matrix(b['mat']).as_quat()
        e.set('quat',' '.join(f'{v:.8f}' for v in (q[3],q[0],q[1],q[2])))
        if total_original_mass>0:
            e.set('mass',f'{total_original_mass/max(1,len(merged)):.7f}')
        b['body'].append(e)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)),exist_ok=True)
    tree.write(output_path,encoding='unicode',xml_declaration=False)
    return {'source':str(input_path),'preview':str(output_path),'original_boxes':original,
            'removed_thin':len(dropped),'merged_pairs':len(merge_events),
            'final_boxes':len(merged),'other_geoms':other,
            'removed_names':[n for b in dropped for n in b['ids']],
            'merge_events':[{'a':list(a),'b':list(b)} for a,b in merge_events]}


def preview_directory(input_dir,output_dir=None,**options):
    input_dir=Path(input_dir)
    output_dir=Path(output_dir) if output_dir else input_dir/'__simplify_preview'
    output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    for src in sorted(input_dir.iterdir()):
        if not src.is_file() or not re.fullmatch(r'atom_\d+\.xml',src.name):
            continue
        result=simplify_mjcf(src,output_dir/src.name,**options)
        results.append(result)
        print(f"[Simplify] {src.name}: {result['original_boxes']} -> {result['final_boxes']} "
              f"(thin={result['removed_thin']}, merge={result['merged_pairs']})")
    import json
    (output_dir/'simplify_report.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'[Simplify] Preview-only output: {output_dir} (originals unchanged)')
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Generate simplified MJCF previews without overwriting original assets')
    p.add_argument('--input',required=True)
    p.add_argument('--output')
    p.add_argument('--thin',type=float,default=0.08)
    p.add_argument('--gap',type=float,default=0.03)
    p.add_argument('--size-tolerance',type=float,default=0.10)
    p.add_argument('--rotation-tolerance',type=float,default=3.0)
    p.add_argument('--max-volume-ratio',type=float,default=1.18)
    p.add_argument('--overlap-ratio',type=float,default=0.70)
    a=p.parse_args()
    preview_directory(a.input,a.output,thin=a.thin,gap=a.gap,size_tolerance=a.size_tolerance,
                      rotation_tolerance=a.rotation_tolerance,max_volume_ratio=a.max_volume_ratio,
                      overlap_ratio=a.overlap_ratio)
