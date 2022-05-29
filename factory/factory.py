class Factory:
    __human_name_to_class__ = dict()

    @classmethod
    def create(cls,
               base,
               human_name,
               params):
        impls = cls.__human_name_to_class__.get(base)
        if impls is None:
            raise Exception(f'No implementations registered for class {base}')
        if human_name not in impls.keys():
            raise Exception(f'No {human_name} name registered among {base} implementations: {impls.keys()}')
        return impls.get(human_name)(**params)

    @classmethod
    def register(cls,
                 base_class,
                 classes: dict):
        dest = cls.__human_name_to_class__.setdefault(base_class, dict())
        for name, impl in classes.items():
            dest[name] = impl


def make_instance(base_class: type,
                  instance_info_supplier):
    if issubclass(type(instance_info_supplier), base_class):
        return instance_info_supplier
    else:
        params = None
        if isinstance(instance_info_supplier, str):
            import json
            if instance_info_supplier.endswith('.json'):
                with open(instance_info_supplier, 'r') as f:
                    params = json.load(f)
            else:
                params = json.loads(instance_info_supplier)
        elif isinstance(instance_info_supplier, dict):
            params = instance_info_supplier
        params = params.copy()
        human_type_name = params.pop("name")
        instance = Factory.create(base_class,
                                  human_type_name,
                                  params)
        assert issubclass(type(instance), base_class)
        return instance
