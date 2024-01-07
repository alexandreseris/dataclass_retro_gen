import io
import copy
from dataclasses import dataclass, field, make_dataclass, is_dataclass, fields
import keyword
import types
from typing import Any, Iterable, Union, cast, get_args, get_origin, Callable, TypeAlias


def guess_json_type(item: Any) -> type | None:
    if item is None:
        return None
    if isinstance(item, dict):
        return dict
    if isinstance(item, list):
        return list
    if isinstance(item, bool):
        return bool
    if isinstance(item, int):
        return int
    if isinstance(item, float):
        return float
    if isinstance(item, str):
        return str
    raise ValueError(f"unknown type {type(item)} for {item}")


def generate_class_name(json_name: str) -> str:
    json_name = json_name[0].upper() + json_name[1:]
    while json_name in keyword.kwlist:
        json_name += "_"
    return json_name


def generate_field_name(json_name: str) -> str:
    json_name = json_name[0].lower() + json_name[1:]
    while json_name in keyword.kwlist:
        json_name += "_"
    return json_name


@dataclass
class Type:
    origin_type: type | None
    args: set["Type"] = field(default_factory=set)

    def __init__(self, origin_type: type | None, *args: "Type") -> None:
        self.origin_type = origin_type
        self.args = set(args)

    def __hash__(self) -> int:
        return hash(self.origin_type) + sum(hash(x) for x in self.args)

    def __lt__(self, obj: object) -> bool:
        if not isinstance(obj, Type):
            raise ValueError(f"Type cannot be compared to {type(obj).__name__}")
        return hash(self.origin_type) > hash(obj.origin_type)

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, Type):
            return False
        if self.origin_type != obj.origin_type:
            return False
        if len(self.args) != len(obj.args):
            return False
        for self_arg, other_arg in zip(sorted(self.args), sorted(obj.args)):
            if self_arg != other_arg:
                return False
        return True

    @classmethod
    def from_python_type(cls, type: type | None):
        origin_type = get_origin(type) or type
        args: list[Type] = []
        for arg in get_args(type):
            args.append(cls.from_python_type(arg))
        return Type(origin_type, *args)

    def is_union(self) -> bool:
        return self.origin_type == Union

    def get_union_arg(self):
        for arg in self.args:
            if arg.is_union():
                return arg
        return None

    def get_first_arg(self):
        for arg in self.args:
            return arg
        raise ValueError("no arg found")

    def generate_union(self, aditionnal_types: Iterable["Type"]) -> None:
        current_self_copy = copy.deepcopy(self)
        self.origin_type = cast(type, Union)
        self.args = set([current_self_copy])
        for arg in aditionnal_types:
            self.args.add(arg)

    def merge(self, other_type: "Type") -> None:
        if self == other_type:
            return
        self_is_union = self.is_union()
        other_type_is_union = other_type.is_union()
        if self_is_union and not other_type_is_union:
            self.args.add(other_type)
            return
        if not self_is_union and other_type_is_union:
            self.generate_union(copy.deepcopy(other_type.args))
            return
        if self.origin_type != other_type.origin_type:
            self.generate_union(copy.deepcopy([other_type]))
            return
        if self.args != other_type.args:
            if len(self.args) > 1 or len(other_type.args) > 1:
                raise ValueError(f"something went terrebly wrong: {self} {other_type}")
            self_union_arg = self.get_union_arg()
            other_type_union_arg = other_type.get_union_arg()
            if self_union_arg and other_type_union_arg:
                for arg in other_type_union_arg.args:
                    self_union_arg.args.add(arg)
                return
            if self_union_arg and not other_type_union_arg:
                for arg in other_type.args:
                    self_union_arg.args.add(arg)
                return
            if not self_union_arg and other_type_union_arg:
                self.get_first_arg().generate_union(other_type_union_arg.args)
                return
            if not self_union_arg and not other_type_union_arg:
                self.get_first_arg().generate_union(other_type.args)

    def to_python_standard_type(self) -> type:
        self.origin_type = cast(type, self.origin_type)
        if not self.args:
            return self.origin_type
        type_alias = types.GenericAlias(self.origin_type, tuple(typ.to_python_standard_type() for typ in self.args))
        return cast(type, type_alias)

    def origin_type_name(self) -> str:
        if self.origin_type is None:
            return "None"
        type_name = self.origin_type.__name__
        if is_dataclass(self.origin_type):
            return f'"{type_name}"'
        return type_name

    def get_dependencies_types(self) -> tuple[set[type], set[type]]:
        if self.origin_type is None:
            return set(), set()
        dataclasses: set[type] = set()
        other_non_builtins: set[type] = set()
        if is_dataclass(self.origin_type):
            dataclasses.add(self.origin_type)
        elif self.origin_type.__module__ != "builtins":
            other_non_builtins.add(self.origin_type)
        for arg in self.args:
            sub_dataclasses, sub_other_non_builtins = arg.get_dependencies_types()
            dataclasses.update(sub_dataclasses)
            other_non_builtins.update(sub_other_non_builtins)
        return dataclasses, other_non_builtins

    def to_string(self, use_typing_union: bool) -> str:
        if not self.args:
            return f"{self.origin_type_name()}"
        if not use_typing_union and self.origin_type is Union:
            return " | ".join([arg.to_string(use_typing_union) for arg in self.args])
        else:
            args_str = ", ".join([arg.to_string(use_typing_union) for arg in self.args])
            return f"{self.origin_type_name()}[{args_str}]"


def get_dataclass_types(dataclass_: type) -> tuple[set[type], set[type]]:
    # this could fail if typing got circular references, but that should never happen
    imports: set[type] = set()
    dataclasses: set[type] = set()
    for field_ in fields(dataclass_):
        type_ = Type.from_python_type(field_.type)
        sub_dataclasses, other_non_builtins = type_.get_dependencies_types()
        imports.update(other_non_builtins)
        dataclasses.update(sub_dataclasses)
    for dataclass_field in dataclasses.copy():
        sub_dataclasses, other_non_builtins = get_dataclass_types(dataclass_field)
        imports.update(other_non_builtins)
        dataclasses.update(sub_dataclasses)
    return dataclasses, imports


def generate_code(dataclasses: set[type], imports: set[type], use_from_import: bool, use_typing_union: bool):
    code_buffer = io.StringIO()
    imports.add(cast(type, dataclass))
    for import_ in sorted(imports, key=lambda x: (x.__module__, x.__name__)):
        if not use_typing_union and import_ is Union:
            continue
        if use_from_import:
            code_buffer.write(f"from {import_.__module__} import {import_.__name__}\n")
        else:
            code_buffer.write(f"import {import_.__module__}\n")
    for dataclass_ in sorted(dataclasses, key=lambda x: (x.__module__, x.__name__)):
        code_buffer.write("@dataclass\n")
        code_buffer.write(f"class {dataclass_.__name__}:\n")
        dataclass_fields = fields(dataclass_)
        if not dataclass_fields:
            code_buffer.write("    pass\n")
            continue
        for field_ in dataclass_fields:
            type_ = Type.from_python_type(field_.type)
            code_buffer.write(f"    {field_.name}: {type_.to_string(use_typing_union)}\n")
    return code_buffer.getvalue()


GUESS_TYPE_SIGNATURE: TypeAlias = Callable[[Any], type | None]


@dataclass
class Dataclass:
    name: str
    fields: dict[str, Type] = field(default_factory=dict)

    def merge(self, dataclass_: "Dataclass") -> None:
        if self.name != dataclass_.name:
            raise ValueError(f"cant merge two Dataclass with different names ({self.name} and {dataclass_.name})")
        for fieldname in self.fields:
            if fieldname in dataclass_.fields:
                self.fields[fieldname].merge(dataclass_.fields[fieldname])
        for fieldname in dataclass_.fields:
            if fieldname not in self.fields:
                dataclass_.fields[fieldname].merge(Type(None))
                self.add_field(fieldname, dataclass_.fields[fieldname], skip_name_generate=True)

    def to_python_standard_dataclass(self):
        return make_dataclass(self.name, {k: v.to_python_standard_type() for k, v in self.fields.items()}.items())

    def add_field(self, name: str, type: Type, skip_name_generate: bool = False) -> None:
        if skip_name_generate:
            fieldname = name
        else:
            fieldname = generate_field_name(name)
            if fieldname != name:
                print(f"WARN - {name} of {self.name} has been aliased to {fieldname}")
        self.fields[fieldname] = type

    @classmethod
    def reduce_dict_list(cls, name: str, data_list: list[dict], guess_type: GUESS_TYPE_SIGNATURE):
        field_dataclass = cls.from_json_dict(name, data_list[0], guess_type=guess_type)
        for item in data_list[1:]:
            field_dataclass.merge(cls.from_json_dict(name, item, guess_type=guess_type))
        return field_dataclass

    @classmethod
    def compute_type(cls, name: str, item: Any, guess_type: GUESS_TYPE_SIGNATURE) -> Type:
        type = guess_type(item)
        if type is not None and issubclass(type, dict):
            field_dataclass = cls.from_json_dict(name, item, guess_type=guess_type)
            first_type = Type(field_dataclass.to_python_standard_dataclass())
        elif type is not None and issubclass(type, list):
            first_type = cls.reduce_list_to_type(name, item, guess_type=guess_type)
        else:
            first_type = Type(type)
        return first_type

    @classmethod
    def reduce_list_to_type(cls, name: str, data_list: list, guess_type: GUESS_TYPE_SIGNATURE):
        first_type = cls.compute_type(name, data_list[0], guess_type=guess_type)
        for item in data_list[1:]:
            first_type.merge(cls.compute_type(name, item, guess_type=guess_type))
        return first_type

    @classmethod
    def from_json_dict(cls, name: str, data: dict, guess_type: GUESS_TYPE_SIGNATURE = guess_json_type):
        dataclass_ = cls(name=generate_class_name(name))

        for field_name, field_value in data.items():
            field_type = guess_type(field_value)

            if field_type is not None and issubclass(field_type, dict):
                field_type = cls.from_json_dict(
                    field_name, field_value, guess_type=guess_type
                ).to_python_standard_dataclass()

            elif field_type is not None and issubclass(field_type, list):
                if field_value:
                    field_type = cls.reduce_list_to_type(field_name, field_value, guess_type).to_python_standard_type()
                    field_type = cast(type, types.GenericAlias(list, field_type))

            dataclass_.add_field(field_name, Type.from_python_type(field_type))
        return dataclass_

    @classmethod
    def from_json_dicts(cls, name: str, data: list[dict], guess_type: GUESS_TYPE_SIGNATURE = guess_json_type):
        if not data:
            raise ValueError("can't generate model from empty list!")
        return cls.reduce_dict_list(name, data, guess_type)

    def generate_class_definition(
        self, buffer: io.TextIOBase, use_from_import: bool = True, use_typing_union: bool = False
    ) -> None:
        std_dataclass = self.to_python_standard_dataclass()
        dataclasses, imports = get_dataclass_types(std_dataclass)
        dataclasses.add(std_dataclass)
        buffer.write(generate_code(dataclasses, imports, use_from_import, use_typing_union))
