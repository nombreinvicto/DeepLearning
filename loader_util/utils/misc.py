import tracemalloc
from decohints import decohints
from functools import wraps
from types import FunctionType
from objsize import get_deep_size
from time import perf_counter


# %% ##################################################################
@decohints
def timed(fn: FunctionType):
    @wraps(fn)
    def inner(*args, **kwargs):
        starttime = perf_counter()
        ret_val = fn(*args, **kwargs)
        endtime = perf_counter()
        elapsed_time = endtime - starttime
        print(f"[INFO] time elapsed: {elapsed_time: 0.3f} secs")
        return ret_val

    return inner


# %% ##################################################################
@decohints
def memprofile(fn: FunctionType) -> FunctionType:
    @wraps(fn)
    def inner(*args, **kwargs):
        tracemalloc.stop()
        tracemalloc.clear_traces()
        tracemalloc.start()
        ret_val = fn(*args, **kwargs)
        stats = tracemalloc.take_snapshot().statistics('lineno')
        bytes_allocated = stats[0].size

        allowed_boundaries = ['KB', 'MB', 'GB', 'TB']
        for i in range(len(allowed_boundaries)):
            q, _ = divmod(bytes_allocated, 1024)
            if q == 0:
                break
            bytes_allocated = bytes_allocated / 1024
        memory_used = f"{bytes_allocated:0.2f} {allowed_boundaries[i - 1]}"
        print(f"[INFO] memory allocated: {memory_used}")
        return ret_val

    return inner


# %% ##################################################################
def obj_size(obj):
    byte_size = get_deep_size(obj)
    allowed_boundaries = ["KB", "MB", "GB", "TB"]

    for i in range(len(allowed_boundaries)):
        q, _ = divmod(byte_size, 1024)
        if q == 0:
            break
        byte_size = byte_size / 1024

    print(f"{byte_size:0.2f} {allowed_boundaries[i - 1]}")
