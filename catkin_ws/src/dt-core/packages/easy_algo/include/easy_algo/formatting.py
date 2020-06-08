import duckietown_utils as dtu
from .algo_db import EasyAlgoFamily


def format_db(db, colorize=True, verbose=False):
    families = list(db.family_name2config.values())
    s = format_families(families, colorize, verbose=verbose)
    return s

def format_families(families, colorize=True, verbose=True):
    if not families:
        s = "No algorithm families found."
        return s
    else:

        table = []
        table.append(['Family name',
                  'interface',
                  'pattern',
                  '# found',
                  'valid',
                  'filename',
                  'description',
                  ])
        for family in families:
            assert isinstance(family, EasyAlgoFamily)
            row = []
            row.append(family.family_name)
            row.append(family.interface)
            row.append(family.instances_pattern)
            if not family.instances:
                row.append('\n(none)')
            else:
                n_valid = len([_ for _ in family.instances.values() if _.valid])
                n_invalid = len(family.instances) - n_valid
                ss = '%s' % len(family.instances)
                if n_invalid:
                    ss += dtu.make_red(' (%d invalid)' % n_invalid)
                row.append(ss)

            if family.valid:
                ss = 'yes'
            else:
                ss = 'no: ' + family.error_if_invalid
            row.append(ss)
            row.append(dtu.friendly_path(family.filename))

            if (not family.valid) and colorize:
                row = dtu.make_row_red(row)

            row.append(family.description.strip())
            table.append(row)

        if not verbose:
            dtu.remove_table_field(table, 'filename')

        s = "Found %d algorithm families:\n\n" % len(families)
        s += dtu.indent(dtu.format_table_plus(table, colspacing=4), '   ')

        return s



def format_instances(family, colorize, verbose=False):
    if not family.instances:
        s = ('No instances files found for family "%s" (pattern = %s).\n\n' %
             (family.family_name, family.instances_pattern))
        return s
    else:
        s = ('Found %d instances of algorithm family "%s":\n' %
             (len(family.instances), family.family_name))
        table = []
        table.append(['Instance name',  'constructor',
                      'parameters', 'description', 'filename'])
        for _ in family.instances.values():
            row = []
            name = _.instance_name
            if (not _.valid) and colorize:
                name = dtu.make_red(name)

            row.append(name)

            row.append(_.constructor)
            row.append(dtu.yaml_dump_pretty(_.parameters))

            row.append(_.description)
            row.append(dtu.friendly_path(_.filename))

            table.append(row)

        if not verbose:
            dtu.remove_table_field(table, 'filename')
            dtu.remove_table_field(table, 'description')
        s += dtu.indent(dtu.format_table_plus(table, colspacing=4), '| ')

        for _ in family.instances.values():
            if not _.valid:
                msg = _.error_if_invalid
                s += dtu.make_red('\n' + dtu.indent(msg, '', _.instance_name + ': '))


        return s
