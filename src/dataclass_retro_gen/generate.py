import functools
import io
import copy
from dataclasses import dataclass, field
import keyword
import types
from typing import Any, Iterable, Union, cast, get_args, get_origin, Callable, TypeAlias, Mapping, Sequence


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


def def_generate_class_name(json_name: str) -> str:
    json_name = json_name[0].upper() + json_name[1:]
    while json_name in keyword.kwlist:
        json_name += "_"
    return json_name


def def_generate_field_name(json_name: str) -> str:
    json_name = json_name[0].lower() + json_name[1:]
    while json_name in keyword.kwlist:
        json_name += "_"
    return json_name


def sort_type_args(self: "Type", other: "Type") -> int:
    if self.origin_type is None:
        return 1
    if other.origin_type is None:
        return -1
    self_origin_type_name = self.origin_type_name()
    other_origin_type_name = other.origin_type_name()
    if self_origin_type_name > other_origin_type_name:
        return 1
    if self_origin_type_name < other_origin_type_name:
        return -1
    return 0


type_args_sorter = functools.cmp_to_key(sort_type_args)


@dataclass
class Type:
    origin_type: "type | Dataclass | None"
    args: set["Type"] = field(default_factory=set)

    def __init__(self, origin_type: "type | Dataclass | None", *args: "Type") -> None:
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
                # only the union type can have several args
                # but the union cases were managed above
                # Mapping is not possible because it should be transformed to dataclass
                # so we should only receive non str Sequence here
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

    def origin_type_name(self, quote_dataclass: bool = True) -> str:
        if self.origin_type is None:
            return "None"
        if isinstance(self.origin_type, Dataclass):
            if quote_dataclass:
                return f'"{self.origin_type._name}"'
            return self.origin_type._name
        return self.origin_type.__name__

    def get_dependencies_types(self) -> "tuple[set[Dataclass], set[type]]":
        if self.origin_type is None:
            return set(), set()
        dataclasses: set[Dataclass] = set()
        other_non_builtins: set[type] = set()
        if isinstance(self.origin_type, Dataclass):
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
        sorted_args = sorted(self.args, key=type_args_sorter)
        if not use_typing_union and self.origin_type is Union:
            return " | ".join([arg.to_string(use_typing_union) for arg in sorted_args])
        else:
            args_str = ", ".join([arg.to_string(use_typing_union) for arg in sorted_args])
            return f"{self.origin_type_name()}[{args_str}]"


@dataclass
class Field:
    name: str
    alias: str | None
    type: Type

    def __hash__(self) -> int:
        return hash(self.name)


def generate_code(dataclasses: set["Dataclass"], imports: set[type], use_from_import: bool, use_typing_union: bool):
    code_buffer = io.StringIO()
    imports.add(cast(type, dataclass))
    for import_ in sorted(imports, key=lambda x: (x.__module__, x.__name__)):
        if not use_typing_union and import_ is Union:
            continue
        if use_from_import:
            code_buffer.write(f"from {import_.__module__} import {import_.__name__}\n")
        else:
            code_buffer.write(f"import {import_.__module__}\n")
    for dataclass_ in sorted(dataclasses, key=lambda x: (-x._depth, x._name)):
        code_buffer.write("\n@dataclass\n")
        code_buffer.write(f"class {dataclass_._name}:\n")
        if not dataclass_._fields:
            code_buffer.write("    pass\n")
            continue
        for field_ in dataclass_._fields.values():
            code_buffer.write(f"    {field_.name}: {field_.type.to_string(use_typing_union)}")
            if field_.alias is not None:
                code_buffer.write(f"  # ALIAS = {field_.alias}")
            code_buffer.write("\n")
    return code_buffer.getvalue()


GUESS_TYPE_SIGNATURE: TypeAlias = Callable[[Any], type | None]
GENERATE_CLASS_NAME_SIGNATURE: TypeAlias = Callable[[str], str]
GENERATE_FIELD_NAME_SIGNATURE: TypeAlias = Callable[[str], str]


@dataclass
class Dataclass:
    _name: str
    _fields: dict[str, Field] = field(default_factory=dict)
    _depth: int = field(default=0)

    def __hash__(self) -> int:
        return hash(self._name)

    def _merge(self, dataclass_: "Dataclass", generate_field_name: GENERATE_FIELD_NAME_SIGNATURE) -> None:
        if self._name != dataclass_._name:
            raise ValueError(f"cant merge two Dataclass with different names ({self._name} and {dataclass_._name})")
        for fieldname in self._fields:
            if fieldname in dataclass_._fields:
                self._fields[fieldname].type.merge(dataclass_._fields[fieldname].type)
        for fieldname in dataclass_._fields:
            if fieldname not in self._fields:
                dataclass_._fields[fieldname].type.merge(Type(None))
                self._add_field(
                    fieldname, dataclass_._fields[fieldname].type, generate_field_name, skip_name_generate=True
                )

    def _add_field(
        self,
        name: str,
        type: Type,
        generate_field_name: GENERATE_FIELD_NAME_SIGNATURE,
        skip_name_generate: bool = False,
    ) -> None:
        alias = None
        if skip_name_generate:
            fieldname = name
        else:
            fieldname = generate_field_name(name)
            if fieldname != name:
                alias = name
        self._fields[fieldname] = Field(fieldname, alias, type)

    @classmethod
    def _compute_type(
        cls,
        name: str,
        item: Any,
        guess_type: GUESS_TYPE_SIGNATURE,
        generate_class_name: GENERATE_CLASS_NAME_SIGNATURE,
        generate_field_name: GENERATE_FIELD_NAME_SIGNATURE,
        classnames: set[str],
        parentname: str | None,
        depth: int,
    ) -> Type:
        type = guess_type(item)
        if type is not None and issubclass(type, Mapping):
            field_dataclass = cls._from_mapping(
                name,
                item,
                guess_type=guess_type,
                classnames=classnames,
                parentname=parentname,
                depth=depth + 1,
                generate_class_name=generate_class_name,
                generate_field_name=generate_field_name,
            )
            first_type = Type(field_dataclass)
        elif type is not None and not issubclass(type, str) and issubclass(type, Sequence):
            first_type = cls._reduce_sequence_to_type(
                name,
                item,
                guess_type=guess_type,
                classnames=classnames,
                parentname=parentname,
                depth=depth + 1,
                generate_class_name=generate_class_name,
                generate_field_name=generate_field_name,
            )
        else:
            first_type = Type(type)
        return first_type

    @classmethod
    def _reduce_sequence_to_type(
        cls,
        name: str,
        data_items: Sequence,
        guess_type: GUESS_TYPE_SIGNATURE,
        generate_class_name: GENERATE_CLASS_NAME_SIGNATURE,
        generate_field_name: GENERATE_FIELD_NAME_SIGNATURE,
        classnames: set[str],
        parentname: str | None,
        depth: int,
    ):
        first_type = cls._compute_type(
            name,
            data_items[0],
            guess_type=guess_type,
            classnames=classnames,
            parentname=parentname,
            depth=depth + 1,
            generate_class_name=generate_class_name,
            generate_field_name=generate_field_name,
        )
        for item in data_items[1:]:
            first_type.merge(
                cls._compute_type(
                    name,
                    item,
                    guess_type=guess_type,
                    classnames=classnames,
                    parentname=parentname,
                    depth=depth + 1,
                    generate_class_name=generate_class_name,
                    generate_field_name=generate_field_name,
                )
            )
        return first_type

    def _get_dataclass_types(self) -> "tuple[set[Dataclass], set[type]]":
        # this will fail if typing got circular references, but that should never happen
        imports: set[type] = set()
        dataclasses: set[Dataclass] = set()
        for field_ in self._fields.values():
            sub_dataclasses, other_non_builtins = field_.type.get_dependencies_types()
            imports.update(other_non_builtins)
            dataclasses.update(sub_dataclasses)
        for dataclass_field in dataclasses.copy():
            sub_dataclasses, other_non_builtins = dataclass_field._get_dataclass_types()
            imports.update(other_non_builtins)
            dataclasses.update(sub_dataclasses)
        return dataclasses, imports

    @classmethod
    def _from_mapping(
        cls,
        name: str,
        data: Mapping[str, Any],
        guess_type: GUESS_TYPE_SIGNATURE,
        generate_class_name: GENERATE_CLASS_NAME_SIGNATURE,
        generate_field_name: GENERATE_FIELD_NAME_SIGNATURE,
        classnames: set[str],
        parentname: "str | None",
        depth: int,
    ):
        if classnames is None:
            classnames = set()
        classname = generate_class_name(name)
        if classname in classnames:
            if parentname is not None:
                classname = f"{parentname}{classname}"
        classnames.add(classname)
        dataclass_ = cls(classname, _depth=depth)

        for field_name, field_value in data.items():
            python_field_type = guess_type(field_value)

            field_type = Type(python_field_type)

            if python_field_type is not None and issubclass(python_field_type, Mapping):
                field_type = Type(
                    cls._from_mapping(
                        field_name,
                        field_value,
                        guess_type=guess_type,
                        generate_class_name=generate_class_name,
                        generate_field_name=generate_field_name,
                        classnames=classnames,
                        parentname=classname,
                        depth=depth + 1,
                    )
                )

            elif (
                python_field_type is not None
                and not issubclass(python_field_type, str)
                and issubclass(python_field_type, Sequence)
            ):
                if field_value:
                    reduced_type = cls._reduce_sequence_to_type(
                        field_name,
                        field_value,
                        guess_type,
                        classnames=classnames,
                        parentname=parentname,
                        depth=depth + 1,
                        generate_class_name=generate_class_name,
                        generate_field_name=generate_field_name,
                    )
                    field_type = Type(Sequence, reduced_type)

            dataclass_._add_field(field_name, field_type, generate_field_name)
        return dataclass_

    @classmethod
    def from_mapping(
        cls,
        name: str,
        data: Mapping[str, Any],
        guess_type: GUESS_TYPE_SIGNATURE = guess_json_type,
        generate_class_name: GENERATE_CLASS_NAME_SIGNATURE = def_generate_class_name,
        generate_field_name: GENERATE_FIELD_NAME_SIGNATURE = def_generate_field_name,
    ):
        return cls._from_mapping(
            name,
            data,
            guess_type,
            classnames=set(),
            parentname=None,
            depth=0,
            generate_class_name=generate_class_name,
            generate_field_name=generate_field_name,
        )

    @classmethod
    def from_mappings(
        cls,
        name: str,
        data: Iterable[Mapping[str, Any]],
        guess_type: GUESS_TYPE_SIGNATURE = guess_json_type,
        generate_class_name: GENERATE_CLASS_NAME_SIGNATURE = def_generate_class_name,
        generate_field_name: GENERATE_FIELD_NAME_SIGNATURE = def_generate_field_name,
    ):
        try:
            data_iterator = iter(data)
        except StopIteration:
            raise ValueError("can't generate model without any data line!")
        first_mapping = next(data_iterator)
        classnames: set[str] = set()
        field_dataclass = cls._from_mapping(
            name,
            first_mapping,
            guess_type=guess_type,
            classnames=classnames,
            parentname=None,
            depth=0,
            generate_class_name=generate_class_name,
            generate_field_name=generate_field_name,
        )
        for item in data_iterator:
            field_dataclass._merge(
                cls._from_mapping(
                    name,
                    item,
                    guess_type=guess_type,
                    classnames=classnames,
                    parentname=None,
                    depth=0,
                    generate_class_name=generate_class_name,
                    generate_field_name=generate_field_name,
                ),
                generate_field_name,
            )
        return field_dataclass

    def generate_class_definition(
        self, buffer: io.TextIOBase, use_from_import: bool = True, use_typing_union: bool = False
    ) -> None:
        dataclasses, imports = self._get_dataclass_types()
        dataclasses.add(self)
        buffer.write(generate_code(dataclasses, imports, use_from_import, use_typing_union))
