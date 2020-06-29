import numpy as np
from easy_algo import get_easy_algo_db
from duckietown_segmaps.maps import SegmentsMap, FAMILY_SEGMAPS, FRAME_TILE
from duckietown_segmaps.transformations import TransformationsInfo
from geometry import SE2_from_xytheta

def compose_maps(grid_spacing, tiles):
    db = get_easy_algo_db()

    tinfo = TransformationsInfo()
    FRAME_GLOBAL = 'global'
    
    partial = [] 
    
    for tile in tiles:
        cell = tile['cell']
        name = tile['name']
        rotation = tile['rotation'] 
        tile_map = db.create_instance(FAMILY_SEGMAPS, name)

        x = cell[0] * grid_spacing
        y = cell[1] * grid_spacing
        theta = np.deg2rad(rotation)
        g = SE2_from_xytheta([x,y, theta])
        tinfo.add_transformation(frame1=FRAME_GLOBAL, frame2=FRAME_TILE, g=g)
        
        tile_map2 = tinfo.transform_map_to_frame(tile_map, FRAME_GLOBAL)
        partial.append(tile_map2)

    result = SegmentsMap.merge(partial)
    return result


