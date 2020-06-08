

'''
    dtrv: [v1]
    size: <size in bytes>
    name: <friendly filename>
    mime: mime type
    hash:
        sha1:
        ipfs:
    urls:
    - file://HOSTNAME/path/to/file
    - file:///ipfs/...
    - http://gateway.ipfs.io/ipfs/IPFS
    - http://ipfs.duckietown.org:8080/ipfs/IPFS
    desc: ''


'''
from collections import OrderedDict
from copy import deepcopy
import os

import duckietown_utils as dtu
from duckietown_utils.test_hash import sha1_for_file_cached
from .ipfs_utils import detect_ipfs, get_ipfs_hash_cached

has_ipfs = detect_ipfs()


class DTR(object):

    def __init__(self, dtrv, size, name, mime, hashes, urls, desc):
        self.dtrv = dtrv
        self.size = size
        self.name = name
        self.mime = mime
        self.hash = hashes
        self.urls = urls
        self.desc = desc

    @staticmethod
    def from_yaml(data0):

        data = deepcopy(data0)
        try:
            urls = data.pop('urls')
            mime = data.pop('mime')
            desc = data.pop('desc')
            size = data.pop('size')
            name = data.pop('name')
            dtrv = data.pop('dtrv')
            hashes = data.pop('hash')
        except KeyError as e:
            msg = 'Could not interpret:'
            msg += '\n\n' + dtu.indent(dtu.yaml_dump(data0), ' >')
            msg += '\n\n' + dtu.indent(str(e), '  ')
            raise Exception(msg)

        return DTR(dtrv, size, name, mime, hashes, urls, desc)


def create_dtr_version_1(filename):
    res = OrderedDict()

    hashes = {}

    if has_ipfs:
        Qm = get_ipfs_hash_cached(filename)
        hashes['ipfs'] = Qm

    with dtu.timeit_wall("hashing %s" % filename, minimum=500):
        hashes['sha1'] = sha1_for_file_cached(filename)

    urls = []

    url = _create_file_uri(filename)
    urls.append(url)

    url = dtu.create_hash_url(filename)
    urls.append(url)

    # noinspection PyUnboundLocalVariable
    if 'ipfs' in hashes:
        url = 'file:///ipfs/%s' % Qm
        urls.append(url)

        url = 'http://gateway.ipfs.io/ipfs/%s' % Qm
        urls.append(url)

        url = 'http://ipfs.duckietown.org:8080/ipfs/%s' % Qm
        urls.append(url)

    name = os.path.basename(filename)
    size = os.path.getsize(filename)
    mime = _mime_for_filename(filename)
    comments = ""

    res['dtrv'] = '1'
    res['size'] = size
    res['name'] = name
    res['mime'] = mime
    res['hash'] = hashes
    res['urls'] = urls
    res['desc'] = comments

    # dtu.logger.debug(dtu.yaml_dump(res))
    return res


mimes = {
    '.bag': 'application/x-bag',
    '.mp4': 'video/mp4',
    '.mov': 'video/mov',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webm': 'video/webm',
}


def _mime_for_filename(filename):
    _, extension = os.path.splitext(filename)
    extension = extension.lower()
    return mimes[extension]


def _create_file_uri(filename):
    if not filename.startswith('/'):
        filename = os.path.join(os.getcwd(), filename)
    uri_prefix = _file_uri_prefix()
    uri = uri_prefix + filename
    return  uri

def _file_uri_prefix():
    import socket
    hostname = socket.gethostname()
    uri = 'file://%s' % (hostname)
    return uri


class NotLocalPath(Exception):
    pass

def get_local_filepath(uri):
    ''' For a path file://hostPATH it returns PATH if it starts with this host name
        otherwise raise ValueError()'''
    uri_prefix = _file_uri_prefix()
    if uri.startswith(uri_prefix):
        return uri[len(uri_prefix):]
    else:
        raise NotLocalPath('Not current host: %s' % uri)

