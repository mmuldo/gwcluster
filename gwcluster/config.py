import yaml
import os
from typing import Any, Optional

DEFAULT_CONFIG_FILE = os.path.join(
    os.getenv("HOME"),
    '.gwcluster.yml'
)

class Config:
    '''
    file containing configuration settings for gwcluster

    see .gwcluster.yml.example for sample config file


    Parameters
    ----------
    filepath : str
        path to config file

    Attributes
    ----------
    settings : dict[str, Any]
        key-value settings
    '''
    def __init__(self, filepath: str = DEFAULT_CONFIG_FILE):
        if os.path.exists(filepath):
            with open(filepath, 'r') as config_file:
                self.settings = yaml.load(config_file, Loader=yaml.Loader)
        else:
            # if no config file, set default settings to nothing
            self.settings = None

    def get(self, *keys: str) -> Optional[Any]:
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
        if not self.settings:
            # if no config file, nothing to return
            return None

        current_value = self.settings
        for key in keys:
            if not isinstance(current_value, dict) or key not in current_value:
                # if key not specified in config, nothing to return
                return None
            current_value = current_value[key]

        return current_value

    def clustering(self, module: str) -> Optional[str]:
        '''
        get the clustering algorithm from configs

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
        clustering = self.get(module, 'clustering')

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

