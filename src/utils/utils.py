

def recursive_merge(target, source):
    """Recursively merge two dictionaries. Filling all missing keys in the target with the values from the source, while keeping existing values."""
    for key, value in source.items():
        if key not in target:
            target[key] = value
        else:
            if isinstance(value, dict):
                target[key] = recursive_merge(target[key], value)
            
    return target