
def init_options(name, bases, dc):
    """
    Metaclass for class creation with configuration options. Collects all
    attributes of the present class and its parents, and adds them to an options
    dictionary.
    """
    options = { }
    for d in [dc] + [b.__dict__ for b in bases]:
        for k,v in d.items():
            if not k.startswith('_') and type(v) in [str, int, float, bool, list]:
                options[k] = v
    dc['options'] = options
    return type(name, bases, dc)
