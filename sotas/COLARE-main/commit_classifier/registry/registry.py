class Registry:
    """A generic registry which registers(maps) a object
    to a list of path(str).

    Usage:
        REGISTRY = Registry("example_registry")

        @REGISTRY.register("a")
        def fa():
            return "a"

        @REGISTRY.register("b")
        class B:
            def __call__(self):
                return "b"

        REGISTRY.keys()
        >>> dict_keys(['a', 'b'])

        REGISTRY.get_callable("a")
        >>> <function fa at 0x000001AA5C10FD30>

        REGISTRY.get_callable("b")
        >>> <__main__.B object at 0x0000015768B7FF70>

        REGISTRY.get_callable("c")
        >>> None
    """

    def __init__(self, name):
        self._name = name
        self._objs = {}

    @property
    def name(self):
        return self.name

    def keys(self):
        return self._objs.keys()

    def register(self, *args):
        for arg in args:
            if not isinstance(arg, str):
                raise RuntimeError(f"register object with type{type(arg)}, expect str")

        def wrapper(cls):
            for arg in args:
                self._objs[arg.lower()] = cls
            return cls

        return wrapper

    def has_obj(self, path: str):
        return path.lower() in self._objs

    def get_obj(self, path: str):
        return self._objs[path.lower()]
