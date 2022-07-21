import yaml
import os
from typing import Any, Optional

CONFIG_FILE_PATH = f'{os.getenv("HOME")}/.gwcluster.yml'

# user config settings
if os.path.exists(CONFIG_FILE_PATH):
    with open(CONFIG_FILE_PATH, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
else:
    config = None

def get_config(*keys: str) -> Optional[Any]:
    '''
    hash config settings in order of passed keys

    Parameters
    ----------
    *keys : str
        order of keys in which to recursively hash config

    Returns
    -------
    Optional[Any]
        the value of the desired config setting if specified in config,
        otherwise None
    '''
    if not config:
        # if no config file, nothing to return
        return None

    current_value = config
    for key in keys:
        if not isinstance(current_value, dict) or key not in current_value:
            # if key not specified in config, nothing to return
            return None
        current_value = current_value[key]

    return current_value

def clustering_default(module: str) -> Optional[str]:
    '''
    get the default clustering algorithm from configs

    Parameters
    ----------
    module : str
        the type of data that the clustering is being performed on, i.e.
        the parent key in configs (e.g. 'seismometer')

    Returns
    -------
    Optional[str]
        the name of the default clustering algorithm if it is specified
        in configs, otherwise None
    '''
    clustering = get_config(module, 'clustering')

    if not clustering:
        # if the 'clustering' key is not specified, nothing to return
        return None

    if isinstance(clustering, str):
        # if clustering key has a str value, return that
        return clustering
    elif isinstance(clustering, dict):
        # otherwise, if a dictionary, return the first key
        return next(iter(clustering.keys()))

    # default to return None
