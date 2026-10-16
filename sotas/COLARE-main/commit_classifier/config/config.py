"""A wraped dict class"""
import sys
import json
import copy
import inspect
import importlib.util as imp_util
from typing import Any, Union
from pathlib import Path
from argparse import Namespace


class Config(dict):
    """A wraped dict class, support accessw value through attributes."""

    def __init__(self, config: Union[Namespace, dict] = None) -> None:
        super().__init__()
        self.namespace = {}
        if config is not None:
            # if config is a Namespace object,
            # keep it for generate python launch command
            if isinstance(config, Namespace):
                config = config.__dict__
                self.namespace = copy.deepcopy(config)

            for key, value in config.items():
                self.add_item(key, value)

    def __str__(self) -> str:
        has_attr = len(set(self.keys())) > 1
        if not has_attr:
            return "{}"
        config_str = "{\n"
        for key, value in self.items():
            if key == "namespace":
                continue
            if isinstance(value, str):
                config_str += f'    "{key}"="{value}"\n'
            else:
                config_str += f'    "{key}"={value}\n'
        config_str += "}"
        return config_str

    def __getattr__(self, __name: str) -> Any:
        try:
            return super().__getitem__(__name)
        except Exception as exception:
            raise KeyError(f"{__name} not in Config") from exception

    def __setattr__(self, __name: str, __value: Any) -> None:
        return super().__setitem__(__name, __value)

    def add_item(self, key, value):
        """Add a key-value pair to this config"""
        value = Config.__transform(value)
        self.__setattr__(key, value)

    @staticmethod
    def __transform(item: Any):
        """transform item into `Config` recursively"""
        if isinstance(item, dict):
            return Config(item)
        elif isinstance(item, (list, tuple)):
            return [Config.__transform(it) for it in item]
        return item

    @staticmethod
    def from_json(jsonfile: Union[str, Path]):
        """Build config from json file.

        Args:
            jsonfile (Union[str, Path]): json file path.

        Return:
            Config: `Config` object from specified json file.
        """
        if isinstance(jsonfile, str):
            filepath = Path(jsonfile).absolute()
        elif isinstance(jsonfile, Path):
            filepath = jsonfile.absolute()
        else:
            raise ValueError(
                f"filename type should be str or Path, but got {type(jsonfile)}"
            )

        with open(filepath, mode="r") as file:
            res = json.load(file)
        return Config(res)

    @staticmethod
    def from_py(pyfile: Union[str, Path]):
        """Build config from python file.

        Args:
            pyfile (Union(str, Path)): python file path.

        Return:
            Config: `Config` object from specified python file.
        """
        if isinstance(pyfile, str):
            filepath = Path(pyfile).absolute()
        elif isinstance(pyfile, Path):
            filepath = pyfile.absolute()
        else:
            raise ValueError(
                f"filename type should be str or Path, but got {type(pyfile)}"
            )

        assert filepath.suffix == ".py", "only python file supported"

        module_name = str(filepath.stem)
        spec = imp_util.spec_from_file_location(module_name, str(filepath))
        module = imp_util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        config = Config()
        for key, value in module.__dict__.items():
            if (
                key.startswith("__")
                or inspect.ismodule(value)
                or inspect.isclass(value)
            ):
                continue
            config.add_item(key, value)
        del sys.modules[module_name]
        return config

    @staticmethod
    def from_file(filename: Union[str, Path]):
        """Build config from file.

        Args:
            pyfile (Union(str, Path)): python file or json file path.

        Return:
            Config: `Config` object from specified file.
        """
        if isinstance(filename, str):
            filepath = Path(filename).absolute()
        elif isinstance(filename, Path):
            filepath = filename.absolute()
        else:
            raise ValueError(
                f"filename type should be str or Path, but got {type(filename)}"
            )

        assert filepath.suffix in (".py", ".json"), "only py and json config supported"
        if filepath.suffix == ".py":
            return Config.from_py(filepath)
        else:
            return Config.from_json(filepath)

    def to_json(self, filename: str):
        """Export config to json file."""
        with open(filename, "w") as jsonfile:
            config = {k: v for k, v in super().items() if k != "namespace"}
            json.dump(config, jsonfile)
