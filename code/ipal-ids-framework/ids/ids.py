import gzip
import hashlib
import json
import sys
from pathlib import Path
import ipal_iids.settings as settings

class MetaIDS:
    _name = None
    _description = ""
    _requires = []
    _metaids_default_settings = {"model-file": "./model"}
    _supports_preprocessor = False

    def __init__(self, name=None):
        self._name = name
        self.settings = settings.idss[self._name]
        self._default_settings = {}
        self._add_default_settings(self._metaids_default_settings)

    def _add_default_settings(self, settings):
        for key, value in settings.items():
            assert key not in self._default_settings
            self._default_settings[key] = value
            if key not in self.settings:
                self.settings[key] = value

    def _open_file(self, filename, mode="r"):
        filename = str(filename)
        if filename is None:
            return None
        elif filename.endswith(".gz"):
            return gzip.open(filename, mode)
        elif filename == "-":
            return sys.stdin
        else:
            return open(filename, mode)

    def _relative_to_config(self, file: str) -> Path:
        config_file = Path(settings.config).resolve()
        file_path = Path(file)
        return (config_file.parent / file_path).resolve()

    def _resolve_model_file_path(self) -> Path:
        if self.settings["model-file"] is None:
            raise Exception("Can't resolve model file since no model file was provided")
        return self._relative_to_config(self.settings["model-file"])

    def _add_msg_hash(self, msg, nbytes=2):
        fingerprint = json.dumps(
            [msg["src"], msg["dest"], msg["protocol"], msg["activity"], msg["type"], msg["length"], msg["data"]]
        )
        fingerprint = fingerprint.encode("utf-8")
        msg["hash"] = int(hashlib.sha1(fingerprint).hexdigest()[: nbytes * 2], 16)

    def requires(self, dataformat):
        if dataformat not in ["train.state", "live.state", "train.ipal", "live.ipal"]:
            settings.logger.critical("Unexpected format requested: {}".format(dataformat))
        return dataformat in self._requires

    def train(self, ipal=None, state=None): raise NotImplementedError
    def new_ipal_msg(self, msg): raise NotImplementedError
    def new_state_msg(self, msg): raise NotImplementedError
    def save_trained_model(self): raise NotImplementedError
    def load_trained_model(self): raise NotImplementedError
    def visualize_model(self): raise NotImplementedError
