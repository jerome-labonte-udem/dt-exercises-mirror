import duckietown_utils as dtu

import numpy as np

from .maps import SegmentsMap, SegMapPoint


__all__ = [
    'TransformationsInfo', 
]

class TransformationsInfo(object):
    """ Keeps track of transformations between poses """
    
    def __init__(self):
        self.t = {}
    
    def add_transformation(self, frame1, frame2, g):
        """ frame2 expressed in frame1 is g in SE(2) """
        if g.shape == (3,3):
            g = dtu.geo.SE3_from_SE2(g)
        elif g.shape == (4,4):
            pass
            
        self.t[(frame1, frame2)] = np.linalg.inv(g)
        self.t[(frame2, frame1)] = g
        self.t[(frame1, frame1)] = np.eye(4)
        self.t[(frame2, frame2)] = np.eye(4)
    
    def transform_point(self, xyz, frame1, frame2):
        """ Transforms point xyz in frame1 to frame2 """
        key = (frame1, frame2)
        
        if not key in self.t:
            msg = 'Could not find transformation %s.' % str(key)
            raise ValueError(msg)
        
        g = self.t[key]
        
        xyzw = np.array([xyz[0],xyz[1],xyz[2],1])
        r = np.dot(g, xyzw)
        res = r[0:3]
        return res
    
    def transform_map_to_frame(self, smap, frame2):
        return _transform_map_to_frame(self, smap, frame2)

@dtu.contract(smap=SegmentsMap)
def _transform_map_to_frame(tinfo, smap, frame2):
    
    def transform_point(p):
        frame1 = p.id_frame
        coords = p.coords
        coords2 = tinfo.transform_point(coords, frame1=frame1, frame2=frame2)
        return SegMapPoint(id_frame=frame2, coords=coords2)
    
    points2 = {}
    for k, v in smap.points.items():
        points2[k] = transform_point(v)
    
    return SegmentsMap(points=points2, segments=smap.segments, faces=smap.faces,
                       constants=smap.constants)

