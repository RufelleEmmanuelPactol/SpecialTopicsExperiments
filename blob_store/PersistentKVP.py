import json
import redis
import os

class PersistentKVP:
    def __init__(self, name):
        self.name = name
        redis_url = os.environ.get('REDIS_URL')
        if not redis_url:
            raise ValueError("REDIS_URL environment variable is not set")
        self.redis_client = redis.from_url(redis_url)

    def __str__(self):
        return json.dumps(self.get_all(), indent=4)

    def __setitem__(self, key, value):
        full_key = f"{self.name}:{key}"
        self.redis_client.set(full_key, json.dumps(value))

    def persist(self):
        # Redis automatically persists data, so this method is not needed
        pass

    def __getitem__(self, key):
        full_key = f"{self.name}:{key}"
        value = self.redis_client.get(full_key)
        if value is None:
            raise KeyError(key)
        return json.loads(value)

    def __contains__(self, key):
        full_key = f"{self.name}:{key}"
        return self.redis_client.exists(full_key)

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        pattern = f"{self.name}:*"
        return [key.decode().split(':', 1)[1] for key in self.redis_client.keys(pattern)]

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def get_all(self):
        return {key: self[key] for key in self.keys()}

    def update(self, other=None, **kwargs):
        if other:
            if hasattr(other, 'keys'):
                for key in other.keys():
                    self[key] = other[key]
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def clear(self):
        pass