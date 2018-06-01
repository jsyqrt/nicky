import os
import os.path as osp

import toml

root_dir = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), '../'))

def join_with_root(path):
    return osp.join(root_dir, path)


class Config:
    def __init__(self, conf):
        self.conf = conf

    def __getattr__(self, name):
        if name not in self.conf:
            return
        if not isinstance(self.conf[name], dict):
            return self.conf[name]
        return Config(self.conf[name])

    def __getitem__(self, key):
        return self.conf[key]


def _get_config():
    def merge_dict(d, d2):
        for k in d2.keys():
            if k in d and isinstance(d[k], dict) and isinstance(d2[k], dict):
                merge_dict(d[k], d2[k])
            else:
                d[k] = d2[k]

    conf = toml.load(join_with_root('nicky/nicky.toml'))
    config_file = os.getenv("NICKY_CONFIG_PATH", "~/.nicky/nicky.toml")
    print('config file path:', config_file)
    user_path = osp.expanduser(config_file)
    if osp.exists(user_path):
        merge_dict(conf, toml.load(user_path))

    return Config(conf)

config = _get_config()

rc_dir = osp.expanduser(config.core.rc_dir)

def join_with_rc(path):
    return osp.join(rc_dir, path)

if __name__ == '__main__':
    print(config)
