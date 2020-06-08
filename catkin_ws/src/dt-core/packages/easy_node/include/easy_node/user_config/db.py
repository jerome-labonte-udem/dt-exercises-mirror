from collections import defaultdict

import yaml
import duckietown_utils as dtu
 
from easy_node.node_description.configuration import load_configuration_for_nodes_in_package, EasyNodeConfig
from easy_node.user_config.get_configuration_files import get_all_configuration_files


class ValidationError(Exception):
    pass


def get_config_db():
    if ConfigDB._singleton is None:
        ConfigDB._singleton = dtu.get_cached('ConfigDB', ConfigDB)
    return ConfigDB._singleton

class ConfigDB():

    _singleton = None

    def __init__(self):
        # Load all configuration
        # filename2contents = look_everywhere_for_config_files()
        
        dtu.logger.debug('Reading configuration files...')
        self.configs = get_all_configuration_files()
        self.package2nodes = {}

        packages = dtu.get_list_of_packages_in_catkin_ws()

        dtu.logger.debug('Reading %d packages configuration...' % len(packages))
        for p in packages:
            self.package2nodes[p] = load_configuration_for_nodes_in_package(p)

        dtu.logger.debug('Validating configuration...')
        
        for i, c in enumerate(self.configs):
            try:
                self.validate_file(c)
                c = c._replace(valid=True)
            except ValidationError as e:
                c = c._replace(valid=False)
                c = c._replace(error_if_invalid=str(e))
                
            self.configs[i] = c 

    def validate_file(self, c):
        # first, check that indeed we have a package by that name
        if not c.package_name in self.package2nodes:
            msg = 'Invalid package "%s".' % c.package_name
            raise ValidationError(msg)
        # check that there is a node by that name
        if not c.node_name in self.package2nodes[c.package_name]:
            msg = 'No node "%s" in package "%s". ' % (c.node_name, c.package_name)
            raise ValidationError(msg)
        # check that all the extends exist
        for cn in c.extends:
            if not self.config_exists(c.package_name, c.node_name, cn):
                msg = 'Referenced config %s/%s/%s does not exist. ' % (c.package_name, c.node_name, cn)
                raise ValidationError(msg)
        # Finally, check that the values correspond to values that we have
        # in the node configuration
        node_config = self.package2nodes[c.package_name][c.node_name]
        assert isinstance(node_config, EasyNodeConfig)
        known = node_config.parameters
        for k in c.values:
            if k not in known:
                msg = 'The parameter "%s" is not known.\nKnown: %s.' % (k, sorted(known))
                raise ValidationError(msg) 
    
    def find(self, package_name, node_name, config_name, date):
        results = []
        for c in self.configs:
            match = (package_name == c.package_name) and \
                    (node_name == c.node_name) and \
                    (config_name == c.config_name)
            if match:
                results.append(c)
        if len(results) > 1:
            raise NotImplementedError('Sort by date')
        if results:
            return results[0]
        else:
            return None
            
    @dtu.contract(package_name='str', node_name='str', config_sequence='list|tuple')
    def resolve(self, package_name, node_name, config_sequence, date=None):
        """ Returns a QueryResult """
        if len(config_sequence) == 0:
            msg = 'Invalid empty config_sequence while querying for %s/%s' % (package_name, node_name)
            raise ValueError(msg) 
        values = {}
        origin = {}
        origin_filename = {}
        
        if not package_name in self.package2nodes:
            msg = ('Could not find package "%s"; I know %s.' % 
                   (package_name, sorted(self.package2nodes)))
            raise dtu.DTConfigException(msg)
        nodes = self.package2nodes[package_name]
        if not node_name in nodes:
            msg = ('Could not find node "%s" in package "%s"; I know %s.' % 
                   (node_name, package_name, sorted(nodes)))
            raise dtu.DTConfigException(msg)
         
        node_config = nodes[node_name]
        all_keys = list(node_config.parameters)

        overridden = defaultdict(lambda: [])
        using = []        
        for config_name in config_sequence:
            if config_name == 'defaults':
                using.append(config_name)
                
                for p in node_config.parameters.values():
         
                    if p.has_default:
                        values[p.name] = p.default
                        origin_filename[p.name] = node_config.filename
                        origin[p.name] = config_name
                 
            else:
                c = self.find(package_name, node_name, config_name, date=date)
                if c is not None:
                    using.append(config_name)
                    
                    for k, v in c.values.items():
                        if k in values:
                            overridden[k].append(origin[k])
                        values[k] = v
                        origin_filename[k] = c.filename
                        origin[k] = config_name 
        
        if not using:
            msg = ('Cannot find any configuration for %s/%s with config sequence %s' % 
                   (package_name, node_name, ":".join(config_sequence)))
            raise dtu.DTConfigException(msg)
         
        return QueryResult(package_name, node_name, config_sequence, all_keys, 
                           values, origin, origin_filename, overridden)

class QueryResult():
    def __init__(self, package_name, node_name, config_sequence, all_keys,
                values, origin, origin_filename, overridden):
        self.all_keys = all_keys
        self.values = values
        self.origin = origin
        self.package_name = package_name
        self.node_name = node_name
        self.config_sequence = config_sequence
        self.origin_filename = origin_filename
        self.overridden = overridden
        assert isinstance(config_sequence, (list, tuple))
    
    def is_complete(self):
        return len(self.all_keys) == len(self.values)
    
    def __str__(self):
        s = 'Configuration result for node `%s` (package `%s`)' % (self.node_name, self.package_name)
        s += '\nThe configuration sequence was %s.' % list(self.config_sequence)
        s += '\nThe following is the list of parameters set and their origin:'
        s +='\n' + dtu.indent(config_summary(self.all_keys, self.values, self.origin), '  ')
        return s
        
@dtu.contract(all_keys='seq(str)', values='dict', origin='dict(str:str)')
def config_summary(all_keys, values, origin):
    table = []
    table.append(['parameter', 'value', 'origin'])
    table.append(['-'*len(_) for _ in table[0]])
    for k in all_keys:
        if k in values:
            v = yaml.dump(values[k])
            v = v.strip()
            if v.endswith('...'):
                v = v[:-3]
            v = v.strip()
            table.append([k,v, dtu.friendly_path(origin[k])])
        else:
            table.append([k, '(unset)', '(not found)'])
    return dtu.format_table_plus(table, 4)

        
        
    

