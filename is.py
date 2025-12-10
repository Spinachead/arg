def is_number(value) -> bool:
    """
    Check if value is a number
    """
    return type(value).__name__ == 'int' or type(value).__name__ == 'float'


def is_string(value) -> bool:
    """
    Check if value is a string
    """
    return type(value).__name__ == 'str'


def is_not_empty_string(value) -> bool:
    """
    Check if value is a non-empty string
    """
    return isinstance(value, str) and len(value) > 0


def is_boolean(value) -> bool:
    """
    Check if value is a boolean
    """
    return type(value).__name__ == 'bool'


def is_function(value) -> bool:
    """
    Check if value is a function
    """
    import inspect
    return inspect.isfunction(value)