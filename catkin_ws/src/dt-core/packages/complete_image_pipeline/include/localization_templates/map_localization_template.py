
from duckietown_segmaps import FAMILY_SEGMAPS, SegmentsMap
from duckietown_segmaps.maps import FRAME_GLOBAL
import duckietown_utils as dtu
from easy_algo import get_easy_algo_db


FAMILY_LOC_TEMPLATES = 'localization_template'

__all__ = [
    'LocalizationTemplate',
    'FAMILY_LOC_TEMPLATES',
]

class LocalizationTemplate(object):
    """ Represents a template on which we can localize. """
    
    def __init__(self, tile_name, dt):
        self.dt = dt
        self.tile_name = tile_name
        self._map = None
        
    @dtu.contract(returns=SegmentsMap)
    def get_map(self):
        if self._map is None:
            # Need to do this here and not in constructor
            # because otherwise there is a loop in EasyAlgo
            db = get_easy_algo_db()
            self._map = db.create_instance(FAMILY_SEGMAPS, self.tile_name)
            
            frames = set(_.id_frame for _ in self._map.points.values())
            if frames != set([FRAME_GLOBAL]):
                msg = ('Expected that all points in the map %r are in the frame %r.' % 
                       (self.tile_name, FRAME_GLOBAL))
                msg += ' These are the frames: %s.' % frames
                raise ValueError(msg)
        return self._map
    
    def get_coords_datatype(self):
        return self.dt 