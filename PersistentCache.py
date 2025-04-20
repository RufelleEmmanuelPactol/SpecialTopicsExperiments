import pickle
import os
import time
from typing import Any, Optional, Dict
from datetime import datetime, timedelta


class PersistentCache:
    def __init__(self, filename: str = 'cache.pkl'):
        """
        Initialize the cache with a filename for persistence.

        Args:
            filename (str): The name of the file to store the cache data
        """
        self.filename = filename
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load the cache from the file if it exists."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'rb') as f:
                    self.cache = pickle.load(f)
                # Clean expired entries on load
                self._cleanup()
            except (pickle.PickleError, EOFError):
                self.cache = {}

    def _save_cache(self) -> None:
        """Save the cache to the file."""
        print('saving cache')
        try:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.cache, f)
        except (pickle.PickleError, IOError) as e:
            print(f"Error saving cache: {e}")

    def _cleanup(self) -> None:
        """Remove expired entries from the cache."""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.cache.items()
            if 'expiry' in value and value['expiry'] <= current_time
        ]
        for key in expired_keys:
            print('Removing expired entries from cache')
            del self.cache[key]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache with an optional TTL in seconds.

        Args:
            key (str): The key to store the value under
            value (Any): The value to store
            ttl (Optional[int]): Time to live in seconds. If None, the entry won't expire
        """
        print('setting')
        cache_entry = {'value': value}
        if ttl is not None:
            cache_entry['expiry'] = time.time() + ttl

        self.cache[key] = cache_entry
        self._save_cache()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.

        Args:
            key (str): The key to retrieve
            default (Any): The default value to return if key not found

        Returns:
            Any: The cached value or default if not found/expired
        """
        self._cleanup()
        self._load_cache()
        print('getting')

        if key in self.cache:
            return self.cache[key]['value']
        return default

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key (str): The key to delete

        Returns:
            bool: True if key was deleted, False if it didn't exist
        """
        if key in self.cache:
            del self.cache[key]
            self._save_cache()
            return True
        return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self.cache = {}
        self._save_cache()

    def get_all(self) -> Dict[str, Any]:
        """
        Get all non-expired values from the cache.

        Returns:
            Dict[str, Any]: Dictionary of all valid cache entries
        """
        self._cleanup()
        return {k: v['value'] for k, v in self.cache.items()}

    def __len__(self) -> int:
        """Return the number of non-expired entries in the cache."""
        self._cleanup()
        return len(self.cache)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache and hasn't expired."""
        self._cleanup()
        self._load_cache()
        print('contains call',  key in self.cache, self.cache)
        return key in self.cache