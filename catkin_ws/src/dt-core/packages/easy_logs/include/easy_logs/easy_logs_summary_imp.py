from collections import defaultdict

from ruamel import yaml

import duckietown_utils as dtu
from easy_logs import get_local_bag_file, NotAvailableLocally
from easy_logs.resource_desc import DTR


def format_logs(logs):
    if not logs:
        s = "No logs found."
        return s
    else:
        s = "Found %d logs.\n" % len(logs)

        table = get_logs_description_table(logs)
        dtu.remove_table_field(table, 'filename')
        dtu.remove_table_field(table, 'topics')
        dtu.remove_table_field(table, 'description')
        dtu.remove_table_field(table, 'hash bag')
        s += dtu.indent(dtu.format_table_plus(table, colspacing=4), '| ')

        counts = defaultdict(lambda: set())
        for l in logs.values():
            for rname, dtr_yaml in l.resources.items():
                counts[rname].add(dtr_yaml['name'])

        s += '\n\nCount of resources: '
        rsort = sorted(counts, key=lambda _: -len(counts[_]))
        for rname in rsort:
            rcount = len(counts[rname])
            s += '\n %3d %s' % (rcount, rname)
            if rcount <= 3:
                s += '  ' + ' '.join(counts[rname])
        return s


def get_logs_description_table(logs, color=True):
    table = []
    table.append(['#', 'Log name',
                  'rc',
                  'description',
                  'bag size',
                  'hash bag',
                  'date',
                  'length',
                  'vehicle name',
                  'filename',
                  'valid',
                  'topics'])
    for i, (_, log) in enumerate(logs.items()):
        row = []
        row.append(i)
        row.append(log.log_name)
        row.append(len(log.resources))
        #        row.append(log.map_name)
        row.append(log.description)
        dtr = DTR.from_yaml(log.resources['bag'])
        bag_size_mb = '%8.1f MB' % (dtr.size / (1024.0 * 1024))
        row.append(bag_size_mb)
        row.append('%s %s \n%s' % (dtr.name, bag_size_mb, dtr.hash['sha1']))
        row.append(log.date)
        if log.length is not None:
            l = '%5.1f s' % log.length
        else:
            l = '(none)'
        row.append(l)
        row.append(log.vehicle)

        try:
            filename = get_local_bag_file(log)
            row.append(dtu.friendly_path(filename))
        except NotAvailableLocally:
            row.append('not local')

        if log.valid:
            sr = 'Yes.'
        else:
            sr = log.error_if_invalid
        row.append(sr)
        if log.bag_info is not None:
            info = yaml.dump(log.bag_info['topics'])
        else:
            info = '(none)'
        if color and not log.valid:
            row = dtu.make_row_red(row)

        row.append(info)
        table.append(row)
    return table
