from datetime import datetime, timedelta
import threading

class SimpleCache:
    """
    简单的内存缓存类，用于存储验证码等临时数据
    """
    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()
    
    def set(self, key, value, expire_in_seconds=300):  # 默认5分钟过期
        """
        设置缓存值
        :param key: 键
        :param value: 值
        :param expire_in_seconds: 过期时间（秒）
        """
        with self._lock:
            self._data[key] = {
                'value': value,
                'expire_time': datetime.now() + timedelta(seconds=expire_in_seconds)
            }
    
    def get(self, key):
        """
        获取缓存值
        :param key: 键
        :return: 值，如果不存在或已过期则返回None
        """
        with self._lock:
            if key in self._data:
                item = self._data[key]
                if datetime.now() < item['expire_time']:
                    return item['value']
                else:
                    # 过期则删除
                    del self._data[key]
            return None
    
    def delete(self, key):
        """
        删除缓存值
        :param key: 键
        """
        with self._lock:
            if key in self._data:
                del self._data[key]

# 全局验证码缓存实例
verification_code_cache = SimpleCache()
captcha_cache = SimpleCache()