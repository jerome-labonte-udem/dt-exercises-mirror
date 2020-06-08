from collections import namedtuple, OrderedDict
import os
from types import NoneType

import duckietown_utils as dtu

__all__ = [
    'EasyNodeConfig',
    'load_configuration',
]

EasyNodeConfig = namedtuple('EasyNodeConfig', 'filename package_name node_type_name description parameters subscriptions contracts publishers')
EasyNodeParameter = namedtuple('EasyNodeParameter', 'name desc type has_default default')
EasyNodeSubscription = namedtuple('EasyNodeSubscription', 'name desc type topic queue_size process latch timeout')
EasyNodePublisher = namedtuple('EasyNodePublisher', 'name desc type topic queue_size latch')

PROCESS_THREADED = 'threaded'
PROCESS_SYNCHRONOUS = 'synchronous'
PROCESS_VALUES = [PROCESS_THREADED, PROCESS_SYNCHRONOUS]



# type = int, bool, float, or None (anything)
DEFAULT_NOT_GIVEN = 'default-not-given'

@dtu.contract(c1=EasyNodeConfig, c2=EasyNodeConfig, returns=EasyNodeConfig)
def merge_configuration(c1, c2):
    """ Merges two configurations. Values in c2 override the ones in c1 """
    parameters = OrderedDict()
    subscriptions = OrderedDict()
    contracts = OrderedDict()
    publishers = OrderedDict()
    for c in [c1, c2]:
        parameters.update(c.parameters)
        subscriptions.update(c.subscriptions)
        contracts.update(c.contracts)
        publishers.update(c.publishers)
    res = EasyNodeConfig(
            filename=c2.filename, # XXX
             package_name=c2.package_name,
             node_type_name=c2.node_type_name,
             description=c2.description,

             parameters=parameters,
             subscriptions=subscriptions,
             contracts=contracts,
             publishers=publishers,
        )
    return res

def load_configuration_baseline():
    """ Get the baseline configuration. """
    c1 = load_configuration_package_node('easy_node', 'easy_node')
    return c1

@dtu.contract(returns=EasyNodeConfig)
def load_configuration_package_node(package_name, node_type_name):
    path = dtu.get_ros_package_path(package_name)
    look_for = '%s.easy_node.yaml' % node_type_name
    found = dtu.locate_files(path, look_for)
    if not found:
        msg = 'Could not find EasyNode configuration %r.' % look_for
        raise dtu.DTConfigException(msg) # XXX

    fn = found[0]
    contents = open(fn).read()
    c = load_configuration(fn, contents)
    c = c._replace(package_name=package_name)
    c = c._replace(node_type_name=node_type_name)

    # Add the common parameters
    if node_type_name != 'easy_node':
        c0 = load_configuration_baseline()
        c = merge_configuration(c0, c)

    return c

@dtu.contract(returns=EasyNodeConfig)
def load_configuration(realpath, contents):
    # TODO: load "version" string
    try:
        try:
            data = dtu.yaml_load(contents)
        except Exception as e:
            msg = 'Could not parse YAML file properly:'
            dtu.raise_wrapped(dtu.DTConfigException, e, msg, compact=True)
        if not isinstance(data, dict):
            msg = 'Expected a dict, got %s.' % type(data).__name__
            raise dtu.DTConfigException(msg)
        try:
            parameters = data.pop('parameters')
            subscriptions = data.pop('subscriptions')
            publishers = data.pop('publishers')
            contracts = data.pop('contracts')
            description = data.pop('description')
        except KeyError as e:
            key = e.args[0]
            msg = 'Invalid configuration: missing field %r.' % (key)
            raise dtu.DTConfigException(msg)

        if not isinstance(description, (str, NoneType)):
            msg = 'Description should be a string, not %s.' % type(description).__name__
            raise dtu.DTConfigException(msg)

        if data:
            msg = 'Spurious fields found: %s' % sorted(data)
            raise dtu.DTConfigException(msg)

        parameters = load_configuration_parameters(parameters)
        subscriptions = load_configuration_subscriptions(subscriptions)
        contracts = load_configuration_contracts(contracts)
        publishers = load_configuration_publishers(publishers)

        return EasyNodeConfig(filename=realpath, parameters=parameters, contracts=contracts,
                              subscriptions=subscriptions, publishers=publishers,
                              package_name=None, description=description,
                              node_type_name=None)
    except dtu.DTConfigException as e:
        msg = 'Invalid configuration at %s: ' % realpath
        dtu.raise_wrapped(dtu.DTConfigException, e, msg, compact=True)

def load_configuration_parameters(data):
    res = OrderedDict()
    for k, v in data.items():
        try:
            check_good_name(k)
            res[k] = load_configuration_parameter(k, v)
        except dtu.DTConfigException as e:
            msg = 'Invalid parameter entry %r:' % k
            dtu.raise_wrapped(dtu.DTConfigException, e, msg, compact=True)
    return res

def load_configuration_subscriptions(data):
    res = OrderedDict()
    for k, v in data.items():
        try:
            check_good_name(k)
            res[k] = load_configuration_subscription(k, v)
        except dtu.DTConfigException as e:
            msg = 'Invalid subscription entry %r:' % k
            dtu.raise_wrapped(dtu.DTConfigException, e, msg, compact=True)
    return res

def load_configuration_publishers(data):
    res = OrderedDict()
    for k, v in data.items():
        try:
            check_good_name(k)
            res[k] = load_configuration_publisher(k, v)
        except dtu.DTConfigException as e:
            msg = 'Invalid publisher entry %r:' % k
            dtu.raise_wrapped(dtu.DTConfigException, e, msg, compact=True)
    return res

def preprocess_desc(d):
    if d is not None:
        return d.strip()

def load_configuration_parameter(name, data):
#     verbose:
#         desc: Whether the node is verbose or not.
#         type: bool
#         default: true
#
    try:
        desc = data.pop('desc', None)
        desc = preprocess_desc(desc)
        type_ = data.pop('type')

        if 'default' in data:
            default = data.pop('default')
            has_default = True
        else:
            default = DEFAULT_NOT_GIVEN
            has_default = False

    except KeyError as e:
        msg = 'Could not find field %r.' % e.args[0]
        raise dtu.DTConfigException(msg)

    if data:
        msg = 'Extra keys: %r' % data
        raise dtu.DTConfigException(msg)

    if not isinstance(desc, (str, NoneType)):
        msg = 'Description should be a string, not %s.' % type(desc).__name__
        raise dtu.DTConfigException(msg)

    type2T = {
        'bool': bool,
        'str': str,
        'int': int,
        'float': float,
        'any': None,
        None: None,
        'dict': dict,
    }

    if not type_ in type2T:
        raise NotImplementedError(type_)
    T = type2T[type_]

    if has_default and default is not None and T is not None:
        default = T(default)

    return EasyNodeParameter(name=name, desc=desc, type=T,
                             has_default=has_default, default=default)

def check_good_name(k):
    # TODO
    pass

def message_class_from_string(s):
    if not '/' in s:
        msg = ''
        msg += 'Invalid message name "%s".\n' % s
        msg += 'I expected that the name of the message is in the format "PACKAGE/MSG".\n '
        msg += 'E.g. "sensor_msgs/Joy" or "duckietown_msgs/BoolStamped".'
        raise dtu.DTConfigException(msg)

    # e.g. "std_msgs/Header"
    i = s.index('/')
    package = s[:i]
    name = s[i+1:]
    symbol = '%s.msg.%s' % (package, name)
    try:
        msgclass = dtu.import_name(symbol)
        return msgclass
    except ValueError as e:
        msg = 'Cannot import type for message "%s" (%s).' % (s, symbol)
        dtu.raise_wrapped(dtu.DTConfigException, e, msg, compact=True)

def load_configuration_subscription(name, data):
#      image:
#         desc: Image to read
#         topic: ~image
#         type: CompressedImage
#         queue_size: 1
    try:
        desc = data.pop('desc', None)
        desc = preprocess_desc(desc)
        latch = data.pop('latch', None)
        timeout = data.pop('timeout', None)
        topic = data.pop('topic')
        type_ = data.pop('type')
        queue_size = data.pop('queue_size', None)
        process = data.pop('process', PROCESS_SYNCHRONOUS)
        if not process in PROCESS_VALUES:
            msg = 'Invalid value of process %r not in %r.' % (process, PROCESS_VALUES)
            raise dtu.DTConfigException(msg)

    except KeyError as e:
        msg = 'Could not find field %r.' % e
        raise dtu.DTConfigException(msg)

    if not isinstance(desc, (str, NoneType)):
        msg = 'Description should be a string, not %s.' % type(desc).__name__
        raise dtu.DTConfigException(msg)

    if data:
        msg = 'Extra keys: %r' % data
        raise dtu.DTConfigException(msg)
    T = message_class_from_string(type_)

    return EasyNodeSubscription(name=name, desc=desc, topic=topic, timeout=timeout,
                                type=T, queue_size=queue_size, latch=latch, process=process)


def load_configuration_publisher(name, data):
    try:
        desc = data.pop('desc', None)
        desc = preprocess_desc(desc)
        latch = data.pop('latch', None)
        topic = data.pop('topic')
        type_ = data.pop('type')
        queue_size = data.pop('queue_size', None)

    except KeyError as e:
        msg = 'Could not find field %r.' % e
        raise dtu.DTConfigException(msg)

    if not isinstance(desc, (str, NoneType)):
        msg = 'Description should be a string, not %s.' % type(desc).__name__
        raise dtu.DTConfigException(msg)

    if data:
        msg = 'Extra keys: %r' % data
        raise dtu.DTConfigException(msg)

    T = message_class_from_string(type_)

    return EasyNodePublisher(name=name, desc=desc,  topic=topic,
                                type=T, queue_size=queue_size, latch=latch)

def load_configuration_contracts(data):
    # TODO
    return {}


def load_configuration_for_nodes_in_package(package_name):
    """
        returns dict node_name -> config
    """
    suffix = '.easy_node.yaml'
    package_dir = dtu.get_ros_package_path(package_name)
    configs = dtu.locate_files(package_dir, '*' + suffix)
    res = {}
    for c in configs:
        node_name = os.path.basename(c).replace(suffix, '')
        res[node_name] = load_configuration_package_node(package_name, node_name)
    return res

@dtu.contract(enc=EasyNodeConfig, returns=str)
def format_enc(enc, descriptions=False):
    s = 'Configuration for node "%s" in package "%s"' % (enc.node_type_name, enc.package_name)
    s += '\n' + '=' * len(s)

    S = ' '*4
    s += '\n\n Parameters\n\n'
    s += dtu.indent(format_enc_parameters(enc, descriptions), S)
    s += '\n\n Subcriptions\n\n'
    s += dtu.indent(format_enc_subscriptions(enc, descriptions), S)
    s += '\n\n Publishers\n\n'
    s += dtu.indent(format_enc_publishers(enc, descriptions), S)
    return s

@dtu.contract(enc=EasyNodeConfig, returns=str)
def format_enc_parameters(enc, descriptions):
    table = []
    table.append(['name',  'type', 'default', 'description',])

    for p in enc.parameters.values():
        if p.desc:
            desc = dtu.wrap_line_length(p.desc, 80)
        else:
            desc = '(none)'
        if p.has_default:
            default = p.default
        else:
            default = '(none)'
        if p.type is None:
            t = '(n/a)'
        else:
            t = p.type.__name__
        table.append([p.name, t, default, desc])
    if not descriptions:
        dtu.remove_table_field(table, 'description')
    return dtu.format_table_plus(table, 2)

@dtu.contract(enc=EasyNodeConfig, returns=str)
def format_enc_subscriptions(enc, descriptions):
    table = []
    table.append(['name',  'type', 'topic', 'options', 'process', 'description',])

    for p in enc.subscriptions.values():
        if p.desc:
            desc = dtu.wrap_line_length(p.desc, 80)
        else:
            desc = '(none)'
        options = []
        if p.queue_size is not None:
            options.append('queue_size = %s' % p.queue_size)
        if p.latch is not None:
            options.append('latch = %s ' %  p.latch)
        if p.timeout is not None:
            options.append('timeout = %s ' %  p.timeout)

        options = '\n'.join(options)
        table.append([p.name, p.type.__name__, p.topic, options, p.process, desc])
    if not descriptions:
        dtu.remove_table_field(table, 'description')
    return dtu.format_table_plus(table, 2)


@dtu.contract(enc=EasyNodeConfig, returns=str)
def format_enc_publishers(enc, descriptions):
    table = []
    table.append(['name',  'type', 'topic', 'options', 'description',])
    
    for p in enc.publishers.values():
        if p.desc:
            desc = dtu.wrap_line_length(p.desc, 80)
        else:
            desc = '(none)'
        options = []
        if p.queue_size is not None:
            options.append('queue_size = %s '% p.queue_size)
        if p.latch is not None:
            options.append('latch = %s' %  p.latch)

        options = '\n'.join(options)
        table.append([p.name, p.type.__name__, p.topic, options, desc])
    if not descriptions:
        dtu.remove_table_field(table, 'description')
    return dtu.format_table_plus(table, 2)
