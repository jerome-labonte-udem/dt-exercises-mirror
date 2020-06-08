from numpy.testing.utils import assert_almost_equal

from duckietown_msgs.msg import Segment
import duckietown_utils as dtu
from geometry import SO2_from_angle
import numpy as np

from .maps import SegmentsMap, FRAME_TILE, SegMapPoint, SegMapFace, SegMapSegment


YELLOW = dtu.ColorConstants.STR_YELLOW
WHITE = dtu.ColorConstants.STR_WHITE
BLACK = dtu.ColorConstants.STR_BLACK
GRAY = dtu.ColorConstants.STR_GRAY
RED = dtu.ColorConstants.STR_RED
GREEN = dtu.ColorConstants.STR_GREEN


def three_way_intersection(tile_size, tile_spacing, width_white):
    constants = {}
    constants['tile_size'] = tile_size
    constants['tile_spacing'] = tile_spacing
    constants['width_white'] = tile_spacing
    points = {}
    segments = []
    faces = []
    extra = (tile_spacing-tile_size)/2
    add_tile(points, faces, segments, tile_size, tile_spacing)
    
    id_frame = FRAME_TILE
    color = WHITE
    x1 = tile_size/2 - width_white
    x2 = tile_size/2
    y1 = -tile_size/2 - extra
    y2 = +tile_size/2 + extra 
    
    _add_rect(points, faces, segments, x1, y1, x2, y2, id_frame, color,
              use_sides_for_loc=[Segment.WHITE, None, Segment.WHITE, None])
    
    
    for a in [0,270]:
        angle = np.deg2rad(a)
        
        add_corner(points, faces, segments, tile_size, extra, width_white, 
                   FRAME_TILE, angle=angle)
        
    data = dict(points=points, segments=segments, faces=faces, constants=constants)

    return SegmentsMap(**data)

@dtu.contract(returns=SegmentsMap)
def get_map_empty_tile(tile_size, tile_spacing, buffer_black):
    constants = {}
    constants['tile_size'] = tile_size
    constants['tile_spacing'] = tile_spacing
    constants['buffer_black'] = buffer_black
    points = {}
    segments = []
    faces = []
    
    add_tile(points, faces, segments, tile_size, tile_spacing)
    
    length = width = tile_size - buffer_black*2 
    x = y = theta = 0
    color = GREEN
    use_sides_for_loc=[None,None,None,None]
    _add_rect_tilted(points, faces, segments, 
                 x, y, theta, length, width, FRAME_TILE, color, 
                 use_sides_for_loc)


    data = dict(points=points, segments=segments, faces=faces, constants=constants)
    return SegmentsMap(**data)
    
@dtu.contract(returns=SegmentsMap)
def empty_tile(tile_size, tile_spacing, width_white):
    """ DT16 Old intersection center """
    constants = {}
    constants['tile_size'] = tile_size
    constants['tile_spacing'] = tile_spacing
    constants['width_white'] = tile_spacing
    points = {}
    segments = []
    faces = []
    
    add_tile(points, faces, segments, tile_size, tile_spacing)
    
    for a in [0,90,180,270]:
        angle = np.deg2rad(a)
        extra = (tile_spacing-tile_size)/2
        add_corner(points, faces, segments, tile_size, extra, width_white, FRAME_TILE, angle=angle)
        
    data = dict(points=points, segments=segments, faces=faces, constants=constants)

    return SegmentsMap(**data)

@dtu.contract(returns=SegmentsMap)
def get_map_intersection_center(tile_size, tile_spacing, width_white, width_red, width_yellow, num_roads):
    
    lane_width = (tile_size - 2*width_white - width_yellow) / 2
    extra = (tile_spacing - tile_size)/2
    
    assert num_roads in [1, 3, 4], num_roads
    
    constants = {}
    constants['tile_size'] = tile_size
    constants['tile_spacing'] = tile_spacing
    constants['width_white'] = tile_spacing # XXX bug
    constants['width_red'] = width_red
    constants['width_yellow'] = width_yellow

    points = {}
    segments = []
    faces = []
    
    add_tile(points, faces, segments, tile_size, tile_spacing)
    
    if num_roads == 1:
        angles = [0]
        
    if num_roads == 3:
        angles = [0, 90, 270]
    if num_roads == 4:
        angles = [0, 90, 180, 270]
    
    id_frame = FRAME_TILE
    
    for a in angles:
        
        # add red
        width = width_red
        length = lane_width
        x = -tile_size/2 + width_red/2
        y = -width_yellow/2 -lane_width/2  
        theta = np.deg2rad(90)
        
        alpha = np.deg2rad(a)
        R = SO2_from_angle(alpha)
        x,y = np.dot(R, [x, y])
        theta = theta + alpha 
        
        color = RED
        use_sides_for_loc=[Segment.RED,None,Segment.RED,None]
        _add_rect_tilted(points, faces, segments, 
                     x, y, theta, length, width, id_frame, color, 
                     use_sides_for_loc)
        
        # add yellow strip
        width = width_red
        length = width_yellow
        x = -tile_size/2 + width_red/2
        y = 0    
        theta = np.deg2rad(90)
        
        alpha = np.deg2rad(a)
        R = SO2_from_angle(alpha)
        x,y = np.dot(R, [x, y])
        theta = theta + alpha 
        
        color = YELLOW
        use_sides_for_loc=[None, None, None, None]
        _add_rect_tilted(points, faces, segments, 
                     x, y, theta, length, width, id_frame, color, 
                     use_sides_for_loc)
        
        # add white corner
        if num_roads == 3 and a == 90:
            # but not in the three-way
            continue
        
        width = width_white
        length = width_white
        
        x = -tile_size/2 + width_white/2
        y = -tile_size/2 + width_white/2
        theta = np.deg2rad(0)
         
        alpha = np.deg2rad(a)
        R = SO2_from_angle(alpha)
        x,y = np.dot(R, [x, y])
        theta = theta + alpha 
         
        color = WHITE
        use_sides_for_loc=[None, None, None, None]
        _add_rect_tilted(points, faces, segments, 
                     x, y, theta, length, width, id_frame, color, 
                     use_sides_for_loc)
         
        width = extra
        length = width_white
        x = -tile_size/2 + width_white/2
        y = -tile_size/2 - width/2
        theta = np.deg2rad(0)
         
        alpha = np.deg2rad(a)
        R = SO2_from_angle(alpha)
        x,y = np.dot(R, [x, y])
        theta = theta + alpha 
         
        color = WHITE
        use_sides_for_loc=[None, None, None, None]
        _add_rect_tilted(points, faces, segments, 
                     x, y, theta, length, width, id_frame, color, 
                     use_sides_for_loc)
        
        width = width_white
        length = extra
        x = -tile_size/2 -length/2 
        y = -tile_size/2 + width_white/2
        theta = np.deg2rad(0)
         
        alpha = np.deg2rad(a)
        R = SO2_from_angle(alpha)
        x,y = np.dot(R, [x, y])
        theta = theta + alpha 
         
        use_sides_for_loc=[None, None, None, None]
        _add_rect_tilted(points, faces, segments, 
                     x, y, theta, length, width, id_frame, color, 
                     use_sides_for_loc)
        
    if num_roads == 3:
        # add white line
        length = tile_size + extra*2
        width = width_white
        x = tile_size/2 - width_white/2
        y = 0
        color = WHITE
        theta = np.deg2rad(90)
        use_sides_for_loc=[Segment.WHITE, None, Segment.WHITE, None]
        _add_rect_tilted(points, faces, segments, 
                     x, y, theta, length, width, id_frame, color, 
                     use_sides_for_loc)
        

    data = dict(points=points, segments=segments, faces=faces, constants=constants)

    return SegmentsMap(**data)


@dtu.contract(returns=SegmentsMap)
def get_map_straight_lane(tile_size, width_yellow, width_white, tile_spacing,
                          gap_len,
                          dash_len, width_red):
    # from inner yellow to inner white
    constants = {}
    constants['tile_size'] = tile_size
    constants['width_yellow'] = width_yellow
    constants['width_white'] = width_white
    constants['dash_len'] = dash_len
    constants['gap_len'] = gap_len
    constants['tile_spacing'] = gap_len

    lane_width = L = (tile_size - 2*width_white - width_yellow) / 2
    constants['lane_width'] = lane_width

    y1 = + width_yellow/2 + L + width_white
    y2 = + width_yellow/2 + L
    y3 = + width_yellow/2
    y4 = - width_yellow/2
    y5 = - width_yellow/2 - L
    y6 = - width_yellow/2 - L - width_white  
    
    assert_almost_equal(width_white + L + width_yellow + L + width_white, tile_size) 
    
    extra = (tile_spacing - tile_size)/2 

    FRAME = FRAME_TILE
    
    S = -tile_size/2 - extra
    D = tile_size/2 + extra
    points = {}
    segments = []
    faces = []
    
    points['p1'] = SegMapPoint(id_frame=FRAME, coords=[S, y1, 0])
    points['q1'] = SegMapPoint(id_frame=FRAME, coords=[D, y1, 0])
    points['p2'] = SegMapPoint(id_frame=FRAME, coords=[S, y2, 0])
    points['q2'] = SegMapPoint(id_frame=FRAME, coords=[D, y2, 0])
    points['p3'] = SegMapPoint(id_frame=FRAME, coords=[S, y3, 0])
    points['q3'] = SegMapPoint(id_frame=FRAME, coords=[D, y3, 0])
    points['p4'] = SegMapPoint(id_frame=FRAME, coords=[S, y4, 0])
    points['q4'] = SegMapPoint(id_frame=FRAME, coords=[D, y4, 0])
    points['p5'] = SegMapPoint(id_frame=FRAME, coords=[S, y5, 0])
    points['q5'] = SegMapPoint(id_frame=FRAME, coords=[D, y5, 0])
    points['p6'] = SegMapPoint(id_frame=FRAME, coords=[S, y6, 0])
    points['q6'] = SegMapPoint(id_frame=FRAME, coords=[D, y6, 0])
    
   
    add_tile(points, faces, segments, tile_size, tile_spacing)
        
    def add_dash(x, length):
        s = len(points)
        pre = 'dash%s_' % s
        points[pre+'t0'] = SegMapPoint(id_frame=FRAME, coords=[x, y3, 0])
        points[pre+'t1'] = SegMapPoint(id_frame=FRAME, coords=[x, y4, 0])
        points[pre+'t2'] = SegMapPoint(id_frame=FRAME, coords=[x+length, y4, 0])
        points[pre+'t3'] = SegMapPoint(id_frame=FRAME, coords=[x+length, y3, 0])
        faces.append(SegMapFace(color=YELLOW, 
                                points=[pre+'t0',pre+'t1',pre+'t2',pre+'t3']))
    
    ngaps = (tile_size) / (gap_len + dash_len) + 1
    if width_red is not None:
        do_not_go_over = tile_size/2
    else:
        do_not_go_over = tile_size/2 + extra
        
    for i in range(int(ngaps)):
        x = i * (gap_len + dash_len) - tile_size/2
        x1 = min(x + dash_len, do_not_go_over)
        length = x1 - x
        if length > 0:
            add_dash(x, length)
    
    segments.append(SegMapSegment(color=Segment.WHITE, points=['q1','p1']))
    segments.append(SegMapSegment(color=Segment.WHITE, points=['p2','q2']))
    segments.append(SegMapSegment(color=Segment.YELLOW, points=['q3','p3']))
    segments.append(SegMapSegment(color=Segment.YELLOW, points=['p4','q4']))
    segments.append(SegMapSegment(color=Segment.WHITE, points=['q5','p5']))
    segments.append(SegMapSegment(color=Segment.WHITE, points=['p6','q6']))
    
    faces.append(SegMapFace(color=WHITE, points=['p1', 'q1', 'q2', 'p2']))
    faces.append(SegMapFace(color=WHITE, points=['p5', 'q5', 'q6', 'p6']))

    if width_red is not None: 
        x1 = tile_size/2 - width_red
        y2 = -width_yellow/2
        y1 = -tile_size/2+width_white
        x2 = tile_size/2 
        
        _add_rect(points, faces, segments, x1, y1, x2, y2, FRAME, RED, 
                  use_sides_for_loc=[Segment.RED,None,Segment.RED,None])
        constants['width_red'] = width_red
        
    data = dict(points=points, segments=segments, faces=faces, constants=constants)

    return SegmentsMap(**data)

@dtu.contract(points='dict', faces='list', id_frame='str', color='str',
              use_sides_for_loc='list[4](None|int)')
def _add_rect(points, faces, segments, x1, y1, x2, y2, id_frame, color,
              use_sides_for_loc):
    assert x2 > x1
    assert y2 > y1
    s = len(points)
    
    pre = '%s_' % s
    names=[pre+'t0',pre+'t1',pre+'t2',pre+'t3']
    points[names[0]] = SegMapPoint(id_frame=id_frame, coords=[x1, y1, 0])
    points[names[1]] = SegMapPoint(id_frame=id_frame, coords=[x1, y2, 0])
    points[names[2]] = SegMapPoint(id_frame=id_frame, coords=[x2, y2, 0])
    points[names[3]] = SegMapPoint(id_frame=id_frame, coords=[x2, y1, 0])
    faces.append(SegMapFace(color=color,points=names))
    
    for i, c in enumerate(use_sides_for_loc):
        if c is not None:
            # note order is important because it should be counterclockwise
            p0 = names[(i+1)%4]
            p1 = names[i]
            segments.append(SegMapSegment(color=c, points=[p0,p1]))

def _add_rect_tilted(points, faces, segments, 
                     x, y, theta, length, width, id_frame, color, 
                     use_sides_for_loc):
    assert length > 0
    assert width > 0
    R = SO2_from_angle(theta)
    p1 = np.array([x, y]) + np.dot(R, [-length/2, -width/2])
    p2 = np.array([x, y]) + np.dot(R, [length/2, -width/2])
    p3 = np.array([x, y]) + np.dot(R, [length/2, width/2])
    p4 = np.array([x, y]) + np.dot(R, [-length/2, width/2])

    coords = reversed([p1, p2, p3, p4])
    __add_rect_by_coords(points, faces, segments, 
                         coords, id_frame, color,
                         use_sides_for_loc)
    
def __add_rect_by_coords(points, faces, segments, 
                         coords, id_frame, color,
                         use_sides_for_loc):
    s = len(points)
    
    pre = '%s_' % s
    names = []
    for i, coord in enumerate(coords):
        name = '%s%s'% (pre, i)
        names.append(name)
        points[name] = SegMapPoint(id_frame=id_frame, 
                                   coords=[coord[0], coord[1], 0])

    faces.append(SegMapFace(color=color,points=names))
    
    for i, c in enumerate(use_sides_for_loc):
        if c is not None:
            # note order is important because it should be counterclockwise
            p0 = names[(i+1)%4]
            p1 = names[i]
            segments.append(SegMapSegment(color=c, points=[p0,p1]))

def add_tile(points, faces, segments, tile_size, tile_spacing):
    """ Add tile bg at 0,0 """
    extra = (tile_spacing - tile_size)/2
    x1 = -tile_size/2 - extra
    x2 = +tile_size/2 + extra
    
    c = BLACK
    # to debug:
    # c = GRAY
    assert_almost_equal(x2-x1, tile_spacing)
    y1 = -tile_size/2 - extra
    y2 = +tile_size/2 + extra
    _add_rect(points, faces, segments, x1, y1, x2, y2, FRAME_TILE, c, 
              use_sides_for_loc=[None,None,None,None])

    x1 = -tile_size/2 
    x2 = +tile_size/2 
    y1 = -tile_size/2 
    y2 = +tile_size/2 
    _add_rect(points, faces, segments, x1, y1, x2, y2, FRAME_TILE, BLACK, 
              use_sides_for_loc=[None,None,None,None])

      
def get_map_curve(tile_size, tile_spacing, width_yellow, 
                        width_white, gap_len, dash_len, direction):
    constants = {}
    constants['tile_size'] = tile_size
    constants['width_yellow'] = width_yellow
    constants['width_white'] = width_white
    constants['dash_len'] = dash_len
    constants['gap_len'] = gap_len
    constants['tile_spacing'] = gap_len

    lane_width = L = (tile_size - 2*width_white - width_yellow) / 2
    constants['lane_width'] = lane_width
    
    assert_almost_equal(width_white + L + width_yellow + L + width_white, tile_size) 
    
    extra = (tile_spacing - tile_size)/2 

    id_frame = FRAME_TILE
    
    points = {}
    segments = []
    faces = []
  
    add_tile(points, faces, segments, tile_size, tile_spacing)
    
    radius = tile_size - width_white/2
    width = width_white
    
    if direction == 'right':
        center = [-tile_size/2, -tile_size/2]
        alpha1 = -np.deg2rad(3)
        alpha2 = np.pi / 2 + np.deg2rad(3)
    else:
        center = [-tile_size/2, +tile_size/2]
        alpha1 = -np.pi / 2 -np.deg2rad(3)
        alpha2 =  + np.deg2rad(3)
        
    colors = [WHITE]
    lengths = [width_white*2]
    detect_color = Segment.WHITE
    
    add_curved(points, faces, segments,id_frame,
               center, radius, alpha1, alpha2, width, 
               colors, lengths, detect_color)

    
    radius = tile_size/2 
        
    width = width_yellow
    colors = [YELLOW, None]
    lengths = [dash_len, gap_len]
    detect_color = Segment.YELLOW
    add_curved(points, faces, segments, id_frame,
               center, radius, alpha1, alpha2, width, 
               colors, lengths, detect_color)
    
    if direction == 'right':
        angle = 0
    else:
        angle = 3*np.pi/2
    add_corner(points, faces, segments, tile_size, extra, width_white, id_frame, angle)
    
    return SegmentsMap(points=points, segments=segments, 
                       faces=faces, constants=constants)

def add_corner(points, faces, segments, tile_size, extra, width_white, id_frame, angle):
    qx = - tile_size/2 - extra
    qy = - tile_size/2 - extra
    S = extra
    
    ax = qx 
    ay = qy + S
    bx = qx + S
    by = qy
    cx = qx + S + width_white
    cy = by
    dx = ax
    dy = qy + S + width_white
    #  d -_
    #  | T
    #  a
    #  S\  
    #  Q b   c

 
    coords = [[ax,ay],[bx,by],[cx,cy],[dx,dy]]
    for i in range(len(coords)):
        R = SO2_from_angle(angle)
        coords[i] = np.dot(R, coords[i])
    color = WHITE
    use_sides_for_loc = [None, None, None, None]
    __add_rect_by_coords(points, faces, segments, 
                         coords, id_frame, color,
                         use_sides_for_loc)
    
@dtu.contract(lengths='list[N]', colors='list[N](str|None)')
def add_curved(points, faces, segments, id_frame, 
               center, radius, 
               alpha1, alpha2, width, colors, lengths, detect_color):
    assert alpha2 > alpha1
    assert len(colors) == len(lengths)
    len_sequence = len(colors)
    
    alpha = alpha1
    for i in range(1000):
        length = lengths[i%len_sequence]
        color = colors[i%len_sequence]
        
        alpha_start = alpha
        effective_radius = radius + width/2
        delta_alpha = length / effective_radius
        alpha_end = alpha_start + delta_alpha
        
        alpha_end = min(alpha_end, alpha2)
#         cut_alpha = alpha_end - alpha_start
        effective_length = (alpha_end-alpha_start) * effective_radius
        if effective_length > 0:
    
            if color is not None:
                
                alpha_mid = (alpha_start + alpha_end)/2
                x = center[0] + np.cos(alpha_mid) * radius
                y = center[1] + np.sin(alpha_mid) * radius
                theta = alpha_mid + np.pi/2
                
                _add_rect_tilted(points, faces, segments, 
                                 x, y, theta, effective_length, width, id_frame, color, 
                                 use_sides_for_loc=[detect_color, None, detect_color, None])
        
        alpha = alpha_end 
        if alpha > alpha2:
            break
        
    