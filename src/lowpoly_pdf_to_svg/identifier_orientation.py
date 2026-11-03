"""Orient piece identifiers using the assembled model's vertical direction."""
import math
import re
import numpy as np
SVG='http://www.w3.org/2000/svg'

def piece_angle(model,piece,point):
    choices=[]
    for face in model['faces']:
        if face['piece']!=piece or face.get('tab'):
            continue
        for k in range(1,len(face['v'])-1):
            ids=[0,k,k+1];flat=np.array([face['flat'][i]for i in ids],float)
            matrix=(flat[1:]-flat[0]).T
            if abs(np.linalg.det(matrix))<1e-9:continue
            vw=np.linalg.solve(matrix,np.array(point)-flat[0]);bary=np.r_[1-vw.sum(),vw]
            world=np.array([model['vertices'][face['v'][i]]for i in ids],float)
            gradient=np.linalg.solve(matrix.T,world[1:,1]-world[0,1])
            if np.linalg.norm(gradient)<1e-6:continue
            angle=math.degrees(math.atan2(gradient[0],-gradient[1]))
            choices.append((sum(max(0,-v)for v in bary),angle))
    return min(choices,key=lambda c:c[0])[1]if choices else 0.0

def orient_identifiers(root,model):
    angles={}
    for group in root.iter('{'+SVG+'}g'):
        piece=group.get('data-piece')
        if not piece:continue
        for text in group.iter('{'+SVG+'}text'):
            identifier=text.get('data-piece-number')
            if identifier is None:continue
            x,y=float(text.get('x')),float(text.get('y'));angle=piece_angle(model,piece,[x,y]);angles[piece]=angle
            text.set('transform',f'rotate({angle:.3f} {x:.3f} {y:.3f})');text.set('data-orientation','assembled-up')
            for path in group.iter('{'+SVG+'}path'):
                if path.get('data-piece-number')!=identifier:continue
                previous=float(path.get('data-frame-angle','0'));delta=math.radians(angle-previous);c,s=math.cos(delta),math.sin(delta)
                def rotate(match):
                    a,b=float(match[1])-x,float(match[2])-y
                    return f'{x+a*c-b*s:.6f},{y+a*s+b*c:.6f}'
                if abs(delta)>1e-10:path.set('d',re.sub(r'([-+\d.eE]+),([-+\d.eE]+)',rotate,path.get('d','')))
                path.attrib.pop('transform',None);path.set('data-frame-angle',str(angle));path.set('data-orientation','assembled-up')
    return angles
