from objsize import get_deep_size


def obj_size(obj):
    byte_size = get_deep_size(obj)
    allowed_boundaries = ["KB", "MB", "GB", "TB"]

    for i in range(len(allowed_boundaries)):
        q, _ = divmod(byte_size, 1024)
        if q == 0:
            break
        byte_size = byte_size / 1024

    print(f"{byte_size:0.2f} {allowed_boundaries[i - 1]}")
