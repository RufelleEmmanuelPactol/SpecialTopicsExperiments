import json
import os

class PersistentKVP:
    def __init__(self, name):
        self.filename = f"{name}.kvp"
        self.data = {}
        self._load()

    def __str__(self):
        return json.dumps(self.data, indent=4)
    def _load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                self.data = json.load(f)
        else:
            self._store()

    def _store(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f)

    def __setitem__(self, key, value):
        self.data[key] = value
        self._store()

    def persist(self):
        self._store()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def update(self, other=None, **kwargs):
        if other:
            self.data.update(other)
        self.data.update(kwargs)
        self._store()

    def clear(self):
        self.data.clear()
        self._store()