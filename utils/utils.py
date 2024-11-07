def get_instance(module, config, *args, **kwargs):
    return getattr(module, config["type"])(*args, **kwargs, **config["args"])
