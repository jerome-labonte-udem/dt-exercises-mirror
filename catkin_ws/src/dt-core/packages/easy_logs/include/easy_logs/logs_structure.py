from collections import namedtuple, OrderedDict

PhysicalLog0 = namedtuple('PhysicalLog',
                          ['log_name',
                           'filename',
                           'map_name',
                           'description',
                           'vehicle',
                           'date', 'length',
                           't0', 't1',  # these are in relative time
                           'size', 'bag_info',
                           'has_camera',
                           'valid', 'error_if_invalid', ])

keys = ['log_name',
        'filename',
        'resources',
        # 'map_name',
        'description',
        'vehicle',
        'date', 'length',
        't0', 't1',  # these are in relative time
        'size', 'bag_info',
        'has_camera',
        'valid', 'error_if_invalid', ]

PhysicalLog = namedtuple('PhysicalLog', keys)


def yaml_from_physical_log(log):
    res = OrderedDict()
    for k in keys:
        res[k] = log.__dict__[k]

    return res


def physical_log_from_yaml(data):
    values = (data[_] for _ in keys)
    return PhysicalLog(*values)
