import json
import redis
import os

class PersistentKVP:
    def __init__(self):
        redis_url = os.environ.get('REDIS_URL')
        if not redis_url:
            raise ValueError("REDIS_URL environment variable is not set")
        self.redis_client = redis.from_url(redis_url)

    def __str__(self):
        return json.dumps(self.get_all(), indent=4)

    def __setitem__(self, key, value):
        self.redis_client.set(key, json.dumps(value))

    def persist(self):
        # Redis automatically persists data, so this method is not needed
        pass

    def __getitem__(self, key):
        value = self.redis_client.get(key)
        if value is None:
            raise KeyError(key)
        return json.loads(value)

    def __contains__(self, key):
        return self.redis_client.exists(key)

    def __len__(self):
        return self.redis_client.dbsize()

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return [key.decode() for key in self.redis_client.keys('*')]

    def values(self):
        return [json.loads(value) for value in self.redis_client.mget(self.keys())]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def get(self, key, default=None):
        value = self.redis_client.get(key)
        return json.loads(value) if value is not None else default

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