import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    print(module, cls)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')

    if 'checkpoint' in config:
        if 'params' in config:
            print(f'Warning: params for {config.target} are ignored')
        kwargs = config.get('kwargs', dict())
        print(f'Loading {config.target} from {config.checkpoint} with {kwargs}')
        return get_obj_from_str(config['target']).load_from_checkpoint(config['checkpoint'], **kwargs)

    if 'params' in config:
        return get_obj_from_str(config['target'])(**config.get('params', dict()))


    raise KeyError('Expected key `params` or `checkpoint` to instantiate.')
