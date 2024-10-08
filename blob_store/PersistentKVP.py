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
        return json.dumps(self.redis_client.hgetall(self.name), indent=4)

    def __setitem__(self, key, value):
        self.redis_client.hset(self.name, key, json.dumps(value))

    def persist(self):
        # Redis automatically persists data, so this method is not needed
        pass

    def __getitem__(self, key):
        value = self.redis_client.hget(self.name, key)
        if value is None:
            raise KeyError(key)
        return json.loads(value)

    def __contains__(self, key):
        return self.redis_client.hexists(self.name, key)

    def __len__(self):
        return self.redis_client.hlen(self.name)

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return [key.decode() for key in self.redis_client.hkeys(self.name)]

    def values(self):
        return [json.loads(value) for value in self.redis_client.hvals(self.name)]

    def items(self):
        return [(key.decode(), json.loads(value)) for key, value in self.redis_client.hgetall(self.name).items()]

    def get(self, key, default=None):
        value = self.redis_client.hget(self.name, key)
        return json.loads(value) if value is not None else default

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