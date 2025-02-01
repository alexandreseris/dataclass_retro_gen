import datetime
import functools
import io
import keyword
from collections.abc import AsyncIterable, Callable, Iterable, Mapping
from collections.abc import Sequence as TypingSequence
from dataclasses import dataclass
from functools import reduce
from inspect import getsource
from io import StringIO
from itertools import chain
from textwrap import dedent, indent
from typing import Any, Generic, Protocol, SupportsIndex, TypeAlias, TypeVar, cast
from typing import Literal as TypingLiteral

from typing_extensions import Self, assert_never

RawScalars: TypeAlias = None | bool | str | float | int
RawValues: TypeAlias = RawScalars | TypingSequence["RawValues"] | Mapping[str, "RawValues"]

NoneType = type(None)
ScalarValue = bool | int | float | str | datetime.date | datetime.datetime | datetime.time | None
UnionMember: TypeAlias = "ScalarType | Structure | Sequence | Literal"
SequenceMember: TypeAlias = "ScalarType | Literal | Union | Structure | Sequence"
FieldType: TypeAlias = "ScalarType | Literal | Union | Structure | Sequence"


class Settings(Protocol):
    """Class used to customize the way parsing and code generation behave"""

    @staticmethod
    def generate_class_name(namepath: "NamePath") -> str:
        """Generate the class name for a mapping using its path

        Args:
            namepath (NamePath): path of the mapping

        Returns:
            str: the class name
        """
        if not namepath._path:
            classname = namepath._root.capitalize()
        else:
            classname = f"{namepath._root.capitalize()}_{'_'.join(x.capitalize() for x in namepath._path)}"
        if keyword.iskeyword(classname):
            classname = f"{classname}_"
        if classname[0].isdigit():
            classname = f"_{classname}"
        return classname

    @staticmethod
    def generate_class_attribute_name(keyname: str) -> str:
        """Generate a class attribute name from the mapping key

        Args:
            keyname (str): the mapping key

        Returns:
            str: the generated attribute name
        """
        if keyword.iskeyword(keyname):
            keyname = f"{keyname}_"
        if keyname[0].isdigit():
            keyname = f"_{keyname}"
        return keyname

    @staticmethod
    def get_sequence_item_path_name() -> str:
        """Specify which name is used for the sequence items. Use when constructing `NamePath`

        Returns:
            str: the item name
        """
        return "item"

    @staticmethod
    def consider_empty_str_as_null() -> bool:
        """Specify if you wish to convert empty string to None

        Returns:
            bool: your desired behavior
        """
        return True

    @staticmethod
    def use_literal(keypath: "NamePath", is_sequence_item: bool, value: RawScalars) -> bool:
        """Specify when to collect literal value instead of types. Called only for scalar values or scalar value in sequence

        Args:
            keypath (NamePath): full path of the value.
            is_sequence_item (bool): whether the value comes from a sequence
            value (RawScalars): raw value

        Returns:
            bool: whether to use literal
        """
        return False

    @staticmethod
    def use_date(keypath: "NamePath", is_sequence_item: bool, value: RawScalars) -> bool:
        """Specify when to try to parse date from value

        Args:
            keypath (NamePath): full path of the value.
            is_sequence_item (bool): whether the value comes from a sequence
            value (RawScalars): raw value

        Returns:
            bool: whether to parse the value
        """
        return False

    @staticmethod
    def parse_date(v: str | int | float | datetime.date) -> datetime.date | None:
        """Customize the way date values are parse. returning None indicates parsing failled.
        This function is copied to the generated code and should **ONLY** use absolute path type anotation (datetime.date instead of date)

        Args:
            v (str | int | float | datetime.date): raw value

        Returns:
            datetime.date | None: the parsed value if succeed else None
        """
        try:
            if isinstance(v, datetime.date):
                return v
            if isinstance(v, str):
                return datetime.date.fromisoformat(v)
            return None
        except ValueError:
            return None

    @staticmethod
    def use_datetime(keypath: "NamePath", is_sequence_item: bool, value: RawScalars) -> bool:
        """Specify when to try to parse datetime from value

        Args:
            keypath (NamePath): path of the parent object's value
            is_sequence_item (bool): whether the value comes from a sequence
            value (RawScalars): raw value

        Returns:
            bool: whether to parse the value
        """
        return False

    @staticmethod
    def parse_datetime(v: str | int | float | datetime.datetime) -> datetime.datetime | None:
        """Customize the way datetime values are parse. returning None indicates parsing failled
        This function is copied to the generated code and should **ONLY** use absolute path type anotation (datetime.datetime instead of date)

        Args:
            v (str | int | float | datetime.datetime): raw value

        Returns:
            datetime.datetime | None: the parsed value if succeed else None
        """
        try:
            if isinstance(v, datetime.datetime):
                return v
            if isinstance(v, str):
                d = datetime.datetime.fromisoformat(v)
                if d.tzinfo is None:
                    return d.replace(tzinfo=datetime.timezone.utc)
                return d
            return None
        except ValueError:
            return None

    @staticmethod
    def use_time(keypath: "NamePath", is_sequence_item: bool, value: RawScalars) -> bool:
        """Specify when to try to parse time from value

        Args:
            keypath (NamePath): path of the parent object's value
            is_sequence_item (bool): whether the value comes from a sequence
            value (RawScalars): raw value

        Returns:
            bool: whether to parse the value
        """
        return False

    @staticmethod
    def parse_time(v: str | datetime.time) -> datetime.time | None:
        """Customize the way time values are parse. returning None indicates parsing failled
        This function is copied to the generated code and should **ONLY** use absolute path type anotation (datetime.time instead of date)

        Args:
            v (str | int | float | datetime.time): raw value

        Returns:
            datetime.time | None: the parsed value if succeed else None
        """
        try:
            if isinstance(v, datetime.time):
                return v
            return datetime.time.fromisoformat(v)
        except ValueError:
            return None


class DefaultSettings(Settings):
    pass


def parse_field(
    keypath: "NamePath", is_sequence_item: bool, value: RawScalars, settings: type[Settings]
) -> ScalarValue:
    if (
        settings.use_date(keypath, is_sequence_item, value)
        and isinstance(value, (str, int, float))
        and (date_value := settings.parse_date(value))
    ):
        return date_value
    if (
        settings.use_datetime(keypath, is_sequence_item, value)
        and isinstance(value, (str, int, float))
        and (datetime_value := settings.parse_datetime(value))
    ):
        return datetime_value
    if (
        settings.use_time(keypath, is_sequence_item, value)
        and isinstance(value, str)
        and (time_value := settings.parse_time(value))
    ):
        return time_value
    return value


_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsAllComparisons(Protocol):
    def __lt__(self, other: _T_contra, /) -> bool:
        ...

    def __gt__(self, other: _T_contra, /) -> bool:
        ...

    def __le__(self, other: _T_contra, /) -> bool:
        ...

    def __ge__(self, other: _T_contra, /) -> bool:
        ...


def build_structure_sorter(settings: type[Settings]) -> Callable[["Structure"], SupportsAllComparisons]:
    def sort_structure(self: "Structure", other: "Structure") -> int:
        if self.depth > other.depth:
            return -1
        if self.depth > other.depth:
            return 1
        self_path_str = settings.generate_class_name(self.structpath)
        other_path_str = settings.generate_class_name(other.structpath)
        if self_path_str > other_path_str:
            return 1
        if self_path_str < other_path_str:
            return -1
        return 0

    structure_sorter = functools.cmp_to_key(sort_structure)
    return structure_sorter


class Comparable(Protocol):
    def __eq__(self, value: object) -> bool:
        ...


C = TypeVar("C", bound=Comparable | None)


class UniqueList(list[C], Generic[C]):
    def __init__(self, it: Iterable[C] | None = None) -> None:
        super().__init__()
        if it is not None:
            for i in it:
                self.append(i)

    def append(self, object: C) -> None:
        if object in self:
            return
        return super().append(object)

    def extend(self, iterable: Iterable[C]) -> None:
        for i in iterable:
            self.append(i)

    def insert(self, index: SupportsIndex, object: C) -> None:
        if object in self:
            return
        return super().insert(index, object)


class NamePath:
    """Describe the path of an object
    eg: for `parse_unstructured_data("somerootname", {"a": [1, 2, 3]}, "b": {"c": 2})`:
    - the path of `a` items is `NamePath("somerootname", ["a", "item"])`.
        the name used for the sequence item is controlled by the setting get_sequence_item_path_name
    - the path of `a.b.c` is `NamePath("somerootname", ["a", "b", "c"])`.
    """

    def __init__(self, root: str, path: tuple[str, ...] = tuple()) -> None:
        self._root = root
        self._path = path

    def __repr__(self) -> str:
        if self._path:
            return f"{self._root} => {'/'.join(self._path)}"
        return self._root

    def __hash__(self) -> int:
        return hash(type(self)) + hash(self._root) + hash(self._path)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return False
        return self._root == value._root and self._path == value._path

    def _add_path(self, name: str) -> "NamePath":
        return NamePath(self._root, self._path + (name,))

    def has_pathpart(self, pathpart: str) -> bool:
        """check if `pathpart` is in the current path"""
        return pathpart in self._path

    def has_any_of_pathparts(self, *pathparts: str) -> bool:
        """check if any of the `pathparts` are in the current path"""
        return any(self.has_pathpart(x) for x in pathparts)

    def has_lastpart(self, lastpart: str) -> bool:
        """check if `lastpart` is the last part of the current path"""
        return len(self._path) > 0 and self._path[-1] == lastpart

    def has_any_of_lastparts(self, *lastparts: str) -> bool:
        """check if any of `lastparts` are the last part of the current path"""
        return any(self.has_lastpart(x) for x in lastparts)

    def is_fullpath(self, fullpath: tuple[str, ...]) -> bool:
        """check if current path match `fullpath`"""
        return self._path == fullpath

    def is_any_of_fullpaths(self, *fullpaths: tuple[str, ...]) -> bool:
        """check if current path match any of the `fullpaths`"""
        return any(self.is_fullpath(x) for x in fullpaths)


class ScalarType:
    def __init__(self, type: type[ScalarValue]) -> None:
        self.type = type

    def __hash__(self) -> int:
        return hash(type(self)) + hash(self.type)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return False
        return self.type == value.type

    def merge(self, other: "ScalarType") -> "Union | None":
        if self.type == other.type:
            return None
        return Union(self, other)

    def is_none(self) -> bool:
        return self.type is NoneType

    def merge_with_field_type(self, other: FieldType) -> FieldType:
        if isinstance(other, ScalarType):
            return self.merge(other) or self
        elif isinstance(other, Literal):
            return Union(self, other)
        elif isinstance(other, Union):
            other.add_type(self)
            return other
        elif isinstance(other, (Structure, Sequence)):
            return Union(self, other)
        else:
            assert_never(other)

    def to_string(self, settings: type[Settings], path: NamePath) -> str:
        if self.type is NoneType:
            return "None"
        if self.type.__module__ == "builtins":
            return self.type.__name__
        return f"{self.type.__module__}.{self.type.__name__}"

    def __repr__(self) -> str:
        return self.to_string(DefaultSettings, NamePath("debug"))

    def to_python_varname(self, settings: type[Settings], valuepath: NamePath) -> str:
        return self.type.__name__

    def get_dependencies_types(self) -> "tuple[UniqueList[Structure], UniqueList[type]]":
        structs: UniqueList[Structure] = UniqueList()
        stdtypes: UniqueList[type] = UniqueList([self.type])
        return (structs, stdtypes)

    def __lt__(self, other: FieldType) -> bool:
        # None goes behind
        if self.is_none():
            return False
        if isinstance(other, ScalarType):
            if other.is_none():
                return True
            return self.type.__name__ < other.type.__name__
        return True

    def sort_types(self) -> None:
        pass

    def get_literal_types(self) -> "list[Literal]":
        return []


class Union:
    def __init__(self, a: UnionMember, b: UnionMember, *others: UnionMember) -> None:
        self.types = UniqueList(chain([a, b], others))

    def __hash__(self) -> int:
        return hash(type(self)) + hash(tuple(self.types))

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return False
        return self.types == value.types

    def has_nullable(self) -> bool:
        for type in self.types:
            if isinstance(type, ScalarType) and type.type is NoneType:
                return True
        return False

    def get_sequence_literal(self) -> "Literal | None":
        for union_type in self.types:
            if isinstance(union_type, Sequence):
                return union_type.get_literal()
        return None

    def get_literal(self) -> "Literal | None":
        for union_type in self.types:
            if isinstance(union_type, Literal):
                return union_type
        return None

    def add_type(self, type: UnionMember) -> None:
        if (
            (self_literal_lookup := self.get_sequence_literal())
            and (isinstance(type, Sequence))
            and (type_literal_lookup := type.get_literal())
        ):
            self_literal_lookup.merge(type_literal_lookup)
            return
        if (self_literal_lookup := self.get_literal()) and (isinstance(type, Literal)):
            self_literal_lookup.merge(type)
            return
        for ind, selftype in enumerate(self.types):
            if selftype == type:
                self.types[ind] = cast(UnionMember, selftype.merge_with_field_type(type))
                return
        self.types.append(type)

    def get_sequence(self) -> "Sequence | None":
        for typ in self.types:
            if isinstance(typ, Sequence):
                return typ
        return None

    def merge(self, other: "Union") -> None:
        self_seq = None
        other_seq = None
        new_types: UniqueList[UnionMember] = UniqueList()
        for typ in self.types:
            if isinstance(typ, Sequence):
                self_seq = typ
            else:
                new_types.append(typ)
        for typ in other.types:
            if isinstance(typ, Sequence):
                other_seq = typ
            else:
                new_types.append(typ)
        if self_seq is not None and other_seq is not None:
            self_seq.merge(other_seq)
            new_types.append(self_seq)
        elif self_seq is not None:
            new_types.append(self_seq)
        elif other_seq is not None:
            new_types.append(other_seq)
        self.types = new_types

    def merge_with_field_type(self, other: FieldType) -> FieldType:
        if isinstance(other, ScalarType):
            self.add_type(other)
            return self
        elif isinstance(other, Literal):
            self.add_type(other)
            return self
        elif isinstance(other, Union):
            self.merge(other)
            return self
        elif isinstance(other, (Structure, Sequence)):
            self.add_type(other)
            return self
        else:
            assert_never(other)

    def to_string(self, settings: type[Settings], path: NamePath) -> str:
        return " | ".join(x.to_string(settings, path) for x in self.types)

    def __repr__(self) -> str:
        return self.to_string(DefaultSettings, NamePath("debug"))

    def to_python_varname(self, settings: type[Settings], valuepath: NamePath) -> str:
        return f"Union_{'__'.join(x.to_python_varname(settings, valuepath) for x in self.types)}"

    def get_dependencies_types(self) -> "tuple[UniqueList[Structure], UniqueList[type]]":
        structs: UniqueList[Structure] = UniqueList()
        stdtypes: UniqueList[type] = UniqueList()
        for type_ in self.types:
            sub_structs, sub_stdtypes = type_.get_dependencies_types()
            structs.extend(sub_structs)
            stdtypes.extend(sub_stdtypes)
        return (structs, stdtypes)

    def __lt__(self, other: FieldType) -> bool:
        if isinstance(other, ScalarType):
            if other.is_none():
                return True
            return False
        elif isinstance(other, Literal):
            return True
        elif isinstance(other, Union):
            raise ValueError("unreachable code")
        elif isinstance(other, (Structure, Sequence)):
            return True
        else:
            assert_never(other)

    def sort_types(self) -> None:
        for t in self.types:
            t.sort_types()
        self.types.sort()

    def get_literal_types(self) -> "list[Literal]":
        types = []
        for t in self.types:
            types.extend(t.get_literal_types())
        return types


class Literal:
    def __init__(self, initial_value: ScalarValue) -> None:
        self.values = UniqueList([initial_value])

    @classmethod
    def from_value(cls, initial_value: ScalarValue) -> "Literal":
        if isinstance(initial_value, float):
            raise ValueError(
                "float type is unsuported in literal :( see https://peps.python.org/pep-0586/#illegal-parameters-for-literal-at-type-check-time "
            )
        return Literal(initial_value)

    def __hash__(self) -> int:
        return hash(type(self)) + hash(tuple(self.values))

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return False
        return self.values == value.values

    def add_value(self, value: ScalarValue) -> "None":
        self.values.append(value)

    def to_type(self) -> "ScalarType | Union":
        types: UniqueList[type] = UniqueList(type(x) for x in self.values)
        if len(types) == 1:
            return ScalarType(types[0])
        else:
            return Union(*[ScalarType(x) for x in types])

    def merge(self, other: "Literal") -> "None":
        self.values.extend(other.values)

    def merge_with_field_type(self, other: FieldType) -> FieldType:
        if isinstance(other, ScalarType):
            return Union(self, other)
        elif isinstance(other, Literal):
            self.merge(other)
            return self
        elif isinstance(other, Union):
            other.add_type(self)
            return other
        elif isinstance(other, Structure):
            raise ValueError("can't use literal with structure data")
        elif isinstance(other, Sequence):
            raise ValueError("can't use literal with sequence data")
        else:
            assert_never(other)

    def to_string(self, settings: type[Settings], path: NamePath) -> str:
        return self.to_python_varname(settings, path)

    def to_python_definition(self) -> str:
        return f"typing.Literal[{', '.join(repr(x) for x in self.values)}]"

    def __repr__(self) -> str:
        return self.to_python_definition()

    def to_python_varname(self, settings: type[Settings], valuepath: NamePath) -> str:
        valuepath_str = settings.generate_class_name(valuepath)
        return f"Literal_{valuepath_str}"

    def get_dependencies_types(self) -> "tuple[UniqueList[Structure], UniqueList[type]]":
        structs: UniqueList[Structure] = UniqueList()
        stdtypes: UniqueList[type] = UniqueList([cast(type, TypingLiteral)])
        return (structs, stdtypes)

    def has_nullable(self) -> bool:
        for value in self.values:
            if value is None:
                return True
        return False

    def __lt__(self, other: FieldType) -> bool:
        if isinstance(other, ScalarType):
            if other.is_none():
                return True
            return False
        elif isinstance(other, Literal):
            raise ValueError("unreachable code")
        elif isinstance(other, Union):
            return False
        elif isinstance(other, Structure):
            return False
        elif isinstance(other, Sequence):
            return False
        else:
            assert_never(other)

    def sort_types(self) -> None:
        self.values.sort(key=lambda x: (x is None, type(x).__name__, x))

    def get_literal_types(self) -> "list[Literal]":
        return [self]


class Sequence:
    def __init__(self, type: SequenceMember) -> None:
        self.type = type

    def __hash__(self) -> int:
        return hash(type(self)) + hash(self.type)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return False
        return self.type == value.type

    def get_literal(self) -> "Literal | None":
        if isinstance(self.type, Literal):
            return self.type
        return None

    def merge(self, other: "Sequence") -> None:
        self.type = self.type.merge_with_field_type(other.type)

    def merge_with_field_type(self, other: FieldType) -> FieldType:
        if isinstance(other, ScalarType):
            return Union(self, other)
        elif isinstance(other, Literal):
            raise ValueError("can't use literal with sequence data")
        elif isinstance(other, Union):
            other.add_type(self)
            return other
        elif isinstance(other, Structure):
            return Union(self, other)
        elif isinstance(other, Sequence):
            self.merge(other)
            return self
        else:
            assert_never(other)

    @classmethod
    def from_seq(
        cls, seqpath: NamePath, seq: TypingSequence[RawValues], depth: int, settings: type[Settings]
    ) -> "Sequence | None":
        ITEM_PATH = settings.get_sequence_item_path_name()
        scalar: Literal | ScalarType | Union | None = None
        strucure: Structure | None = None
        sequence: Sequence | None = None
        for value in seq:
            if isinstance(value, Mapping):
                sub_structure = Structure.from_mapping(seqpath._add_path(ITEM_PATH), value, depth, settings)
                if strucure is None:
                    strucure = sub_structure
                else:
                    strucure.merge(sub_structure)
            elif isinstance(value, TypingSequence) and not isinstance(value, str):
                sub_sequence = Sequence.from_seq(seqpath._add_path(ITEM_PATH), value, depth, settings)
                if sequence is None:
                    sequence = sub_sequence
                elif sub_sequence is not None:
                    sequence.merge(sub_sequence)
            else:
                conv_value = parse_field(seqpath, True, value, settings)
                if settings.consider_empty_str_as_null() and isinstance(value, str) and value == "":
                    conv_value = None
                if settings.use_literal(seqpath, True, value):
                    sub_scalar_literal = Literal.from_value(conv_value)
                    if scalar is None:
                        scalar = sub_scalar_literal
                    else:
                        scalar = scalar.merge_with_field_type(sub_scalar_literal)  # type: ignore[assignment]
                else:
                    sub_scalar = ScalarType(type(conv_value))
                    if scalar is None:
                        scalar = sub_scalar
                    else:
                        scalar = scalar.merge_with_field_type(sub_scalar)  # type: ignore[assignment]
        found_types: list[Literal | ScalarType | Union | Sequence | Structure] = []
        if scalar:
            found_types.append(scalar)
        if strucure:
            found_types.append(strucure)
        if sequence:
            found_types.append(sequence)
        # empty sequence, TODO: should handle that better
        if not found_types:
            return None
        return Sequence(reduce(lambda x, y: x.merge_with_field_type(y), found_types))

    def to_string(self, settings: type[Settings], path: NamePath) -> str:
        return f"collections.abc.Sequence[{self.type.to_string(settings, path)}]"

    def __repr__(self) -> str:
        return self.to_string(DefaultSettings, NamePath("debug"))

    def to_python_varname(self, settings: type[Settings], valuepath: NamePath) -> str:
        return f"Sequence_{self.type.to_python_varname(settings, valuepath)}"

    def get_dependencies_types(self) -> "tuple[UniqueList[Structure], UniqueList[type]]":
        structs: UniqueList[Structure] = UniqueList()
        stdtypes: UniqueList[type] = UniqueList([cast(type, TypingSequence)])
        sub_structs, sub_stdtypes = self.type.get_dependencies_types()
        structs.extend(sub_structs)
        stdtypes.extend(sub_stdtypes)
        return (structs, stdtypes)

    def __lt__(self, other: FieldType) -> bool:
        if isinstance(other, ScalarType):
            if other.is_none():
                return True
            return False
        elif isinstance(other, Literal):
            return True
        elif isinstance(other, Union):
            return True
        elif isinstance(other, Structure):
            return False
        elif isinstance(other, Sequence):
            raise ValueError("unreachable code")
        else:
            assert_never(other)

    def sort_types(self) -> None:
        self.type.sort_types()

    def get_literal_types(self) -> "list[Literal]":
        return self.type.get_literal_types()


class Field:
    def __init__(self, name: str, type: FieldType) -> None:
        self.name = name
        self.type = type

    def __hash__(self) -> int:
        return hash(type(self)) + hash(self.name)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return False
        return self.name == value.name

    def add_null(self) -> None:
        if isinstance(self.type, Literal):
            self.type.add_value(None)
            return
        self.type = self.type.merge_with_field_type(ScalarType(NoneType))

    def is_nullable(self) -> bool:
        if isinstance(self.type, ScalarType) and self.type is NoneType:
            return True
        if isinstance(self.type, Union) and self.type.has_nullable():
            return True
        return False

    def __repr__(self) -> str:
        return f"{self.name}: {repr(self.type)}"


class Structure:
    def __init__(self, structpath: NamePath, fields: dict[str, Field], depth: int) -> None:
        self.structpath = structpath
        self.fields = fields
        self.depth = depth

    def __hash__(self) -> int:
        return hash(type(self)) + hash(self.structpath)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return False
        return self.structpath == value.structpath

    @classmethod
    def from_mapping(
        cls, structpath: NamePath, mapping: Mapping[str, RawValues], depth: int, settings: type[Settings]
    ) -> "Structure":
        fields: dict[str, Field] = {}
        for key, value in mapping.items():
            keypath = structpath._add_path(key)
            if isinstance(value, Mapping):
                fields[key] = Field(key, Structure.from_mapping(keypath, value, depth + 1, settings))
            elif isinstance(value, TypingSequence) and not isinstance(value, str):
                seq = Sequence.from_seq(keypath, value, depth + 1, settings)
                if seq is None:
                    continue
                fields[key] = Field(key, seq)
            else:
                conv_value = parse_field(keypath, False, value, settings)
                if settings.consider_empty_str_as_null() and isinstance(value, str) and value == "":
                    conv_value = None
                if settings.use_literal(keypath, False, value):
                    fields[key] = Field(key, Literal.from_value(conv_value))
                else:
                    fields[key] = Field(key, ScalarType(type(conv_value)))
        return cls(structpath, fields, depth)

    def merge(self, other: "Structure") -> None:
        for key in self.fields:
            # merge common keys
            if key in other.fields:
                new_type = self.fields[key].type.merge_with_field_type(other.fields[key].type)
                self.fields[key].type = new_type
            # self has key but not other
            else:
                self.fields[key].add_null()
        for key in other.fields:
            # other has key but not self
            if key not in self.fields:
                other_field = other.fields[key]
                other_field.add_null()
                self.fields[key] = other_field

    def merge_with_field_type(self, other: FieldType) -> FieldType:
        if isinstance(other, ScalarType):
            return Union(self, other)
        elif isinstance(other, Literal):
            raise ValueError("can't use literal with structure data")
        elif isinstance(other, Union):
            other.add_type(self)
            return other
        elif isinstance(other, Structure):
            self.merge(other)
            return self
        elif isinstance(other, Sequence):
            return Union(self, other)
        else:
            assert_never(other)

    def to_string(self, settings: type[Settings], path: NamePath) -> str:
        return settings.generate_class_name(self.structpath)

    def __repr__(self) -> str:
        return self.to_string(DefaultSettings, NamePath("debug"))

    def to_python_varname(self, settings: type[Settings], valuepath: NamePath) -> str:
        return settings.generate_class_name(self.structpath)

    def get_dependencies_types(self) -> "tuple[UniqueList[Structure], UniqueList[type]]":
        structs: UniqueList[Structure] = UniqueList([self])
        stdtypes: UniqueList[type] = UniqueList()
        for field in self.fields.values():
            sub_structs, sub_stdtypes = field.type.get_dependencies_types()
            structs.extend(sub_structs)
            stdtypes.extend(sub_stdtypes)
        return (structs, stdtypes)

    def __lt__(self, other: FieldType) -> bool:
        if isinstance(other, ScalarType):
            if other.is_none():
                return True
            return False
        elif isinstance(other, Literal):
            return True
        elif isinstance(other, Union):
            return True
        elif isinstance(other, Structure):
            raise ValueError("unreachable code")
        elif isinstance(other, Sequence):
            return True
        else:
            assert_never(other)

    def sort_types(self) -> None:
        for t in self.fields.values():
            t.type.sort_types()

    def get_literal_types(self) -> "list[Literal]":
        return []


class ParseResult:
    def __init__(
        self,
        settings: type[Settings],
        root_structure: Structure,
    ):
        self._settings = settings
        self._root_structure = root_structure

    def write_definitions(self, code_buffer: io.TextIOBase, generate_from_mapping_methods: bool = False) -> None:
        """Write the definitions of the classes generated by parse_unstructured_data

        Args:
            code_buffer (io.TextIOBase): a text buffer used to write the code into
            generate_from_mapping_methods (bool, optional): whether to generate `from_mapping` class method for the classes generated. Defaults to False.
        """
        structures, imports = self._root_structure.get_dependencies_types()
        structures.append(self._root_structure)
        imports.append(cast(type, dataclass))
        if generate_from_mapping_methods:
            for t in (Mapping, Any, datetime.date, datetime.datetime, datetime.time, cast, Self):
                imports.append(cast(type, t))
        import_modules: UniqueList[str] = UniqueList()
        for import_ in sorted(imports, key=lambda x: (x.__module__, x.__name__)):
            if import_.__module__ == "builtins":
                continue
            import_modules.append(import_.__module__)
        has_imported_something = False
        for module in import_modules:
            code_buffer.write(f"import {module}\n")
            has_imported_something = True
        if has_imported_something:
            code_buffer.write("\n\n")
        generate_extract_functions_strs: UniqueList[str] = UniqueList()
        for struct in sorted(structures, key=build_structure_sorter(self._settings)):
            struct_code_buffer = StringIO()
            struct_code_buffer.write("@dataclasses.dataclass\n")
            classname = struct.to_string(self._settings, struct.structpath)
            struct_code_buffer.write(f"class {classname}:\n")
            if not struct.fields:
                struct_code_buffer.write("    pass\n\n\n")
                continue
            for field in struct.fields.values():
                field.type.sort_types()
                attrname = self._settings.generate_class_attribute_name(field.name)
                field_path = struct.structpath._add_path(attrname)
                field_type_str = field.type.to_string(self._settings, field_path).replace('"', '\\"')
                for lit in field.type.get_literal_types():
                    lit_name = lit.to_string(self._settings, field_path)
                    lit_def = lit.to_python_definition()
                    code_buffer.write(f"{lit_name}: typing.TypeAlias = {lit_def}\n")
                mapping_key = ""
                if field.name != attrname:
                    field_name_escaped = repr(field.name).replace('"', '\\"')
                    mapping_key = f'{{"mapping_key": "{field_name_escaped}"}}'
                struct_code_buffer.write(f'    {attrname}: "{field_type_str}" = dataclasses.field({mapping_key})\n')
            if not generate_from_mapping_methods:
                struct_code_buffer.write("\n\n")
                struct_code_buffer.flush()
                struct_code_buffer.seek(0)
                code_buffer.write(struct_code_buffer.read())
                continue

            field_indent = "            "

            def generate_literal_extract_def(literal: Literal, valuepath: NamePath) -> str:
                function_name = f"extract_{literal.to_python_varname(self._settings, valuepath)}"
                type_str = literal.to_string(self._settings, valuepath)
                literal_values = literal.values
                has_datetime = False
                has_date = False
                has_time = False
                for literal_value in literal_values:
                    if isinstance(literal_value, datetime.datetime):
                        has_datetime = True
                    elif isinstance(literal_value, datetime.date):
                        has_date = True
                    elif isinstance(literal_value, datetime.time):
                        has_time = True
                additional_converts = []
                if has_datetime:
                    additional_converts.append(
                        f"""\
dt_value = parse_datetime(value)
if dt_value is not None and dt_value in literal_values:
    return typing.cast({type_str}, value)
"""
                    )
                if has_date:
                    additional_converts.append(
                        f"""\
d_value = parse_date(value)
if d_value is not None and d_value in literal_values:
    return typing.cast({type_str}, value)
"""
                    )
                if has_time:
                    additional_converts.append(
                        f"""\
t_value = parse_time(value)
if t_value is not None and t_value in literal_values:
    return typing.cast({type_str}, value)
"""
                    )
                additional_converts_str = "".join(indent(x, " " * 4) for x in additional_converts)
                generate_extract_functions_strs.append(
                    f"""\
def {function_name}(value: typing.Any) -> "{type_str}":
    literal_values = {repr(literal_values)}
    if value in literal_values:
        return typing.cast({type_str}, value)
{additional_converts_str}
    raise ValueError(f"unexpected value: {{value}}")
"""
                )
                return function_name

            def generate_sequence_extract_def(sequence: Sequence, valuepath: NamePath) -> str:
                function_name = f"extract_{sequence.to_python_varname(self._settings, valuepath)}"
                type_str = sequence.to_string(self._settings, valuepath)

                if isinstance(sequence.type, ScalarType):
                    if sequence.type.is_none():
                        generate_extract_functions_strs.append(
                            dedent(
                                f"""\
                                def {function_name}(value: typing.Any, check_for_remaining_keys: bool) -> "{type_str}":
                                    if not isinstance(value, typing.Sequence):
                                        raise ValueError(f"unepected type: {{type(value).__name__}}")
                                    for x in value:
                                        if x is not None:
                                            raise ValueError(f"unepected type: {{type(value).__name__}}")
                                    return cast(typing.Sequence[None], value)
                                """
                            )
                        )
                        return function_name
                    else:
                        inner_function_type_extract = (
                            f"extract_{sequence.type.to_python_varname(self._settings, valuepath)}"
                        )
                        function_call = f"{inner_function_type_extract}(x)"
                elif isinstance(sequence.type, Literal):
                    inner_function_type_extract = generate_literal_extract_def(sequence.type, valuepath)
                    function_call = f"{inner_function_type_extract}(x)"
                elif isinstance(sequence.type, Sequence):
                    inner_function_type_extract = generate_sequence_extract_def(sequence.type, valuepath)
                    function_call = f"{inner_function_type_extract}(x, check_for_remaining_keys)"
                elif isinstance(sequence.type, Structure):
                    inner_function_type_extract = generate_structure_extract_def(sequence.type, valuepath)
                    function_call = f"{inner_function_type_extract}(x, check_for_remaining_keys)"
                elif isinstance(sequence.type, Union):
                    inner_function_type_extract = generate_union_extract_def(sequence.type, valuepath)
                    function_call = f"{inner_function_type_extract}(x, check_for_remaining_keys)"
                else:
                    assert_never(sequence.type)

                generate_extract_functions_strs.append(
                    dedent(
                        f"""\
                        def {function_name}(value: typing.Any, check_for_remaining_keys: bool) -> "{type_str}":
                            if not isinstance(value, typing.Sequence):
                                raise ValueError(f"unepected type: {{type(value).__name__}}")
                            return [{function_call} for x in value]
                        """
                    )
                )
                return function_name

            def generate_structure_extract_def(structure: Structure, valuepath: NamePath) -> str:
                function_name = f"extract_{structure.to_python_varname(self._settings, valuepath)}"
                type_str = structure.to_string(self._settings, valuepath)
                generate_extract_functions_strs.append(
                    dedent(
                        f"""\
                        def {function_name}(value: typing.Any, check_for_remaining_keys: bool) -> "{type_str}":
                            if not isinstance(value, typing.MutableMapping):
                                raise ValueError(f"unepected type: {{type(value).__name__}}")
                            return {type_str}.from_mapping(value, check_for_remaining_keys)
                        """
                    )
                )
                return function_name

            def generate_union_extract_def(union: Union, valuepath: NamePath) -> str:
                function_name = f"extract_{union.to_python_varname(self._settings, valuepath)}"
                none_expr = """\
if value is None:
    return None
"""
                sub_function_expr = """\
try:
    return {function_call}
except ValueError:
    pass
"""
                type_str = union.to_string(self._settings, valuepath)
                convert_lines: list[str] = []
                has_None = False
                for type in union.types:
                    if isinstance(type, ScalarType):
                        if type.is_none():
                            has_None = True
                            continue
                        else:
                            sub_function_name = f"extract_{type.to_python_varname(self._settings, valuepath)}"
                            function_call = f"{sub_function_name}(value)"
                    elif isinstance(type, Literal):
                        sub_function_name = generate_literal_extract_def(type, valuepath)
                        function_call = f"{sub_function_name}(value)"
                    elif isinstance(type, Sequence):
                        sub_function_name = generate_sequence_extract_def(type, valuepath)
                        function_call = f"{sub_function_name}(value, check_for_remaining_keys)"
                    elif isinstance(type, Structure):
                        sub_function_name = generate_structure_extract_def(type, valuepath)
                        function_call = f"{sub_function_name}(value, check_for_remaining_keys)"
                    elif isinstance(type, Union):
                        sub_function_name = generate_union_extract_def(type)
                        function_call = f"{sub_function_name}(value, check_for_remaining_keys)"
                    else:
                        assert_never(type)
                    convert_lines.append(sub_function_expr.format(function_call=function_call))
                if has_None:
                    convert_lines.insert(0, none_expr)
                convert_lines_expr_strs = "".join(indent(x, " " * 4) for x in convert_lines)
                generate_extract_functions_strs.append(
                    f"""\
def {function_name}(value: typing.Any, check_for_remaining_keys: bool) -> "{type_str}":
{convert_lines_expr_strs}
    raise ValueError(f"unepected type: {{type(value).__name__}}")
"""
                )
                return function_name

            struct_code_buffer.write("    @classmethod\n")
            struct_code_buffer.write(
                "    def from_mapping(cls, data: typing.MutableMapping[str, typing.Any], check_for_remaining_keys: bool = True) -> typing_extensions.Self:\n"
            )
            struct_code_buffer.write("        obj = cls(\n")
            for field in struct.fields.values():
                attrname = self._settings.generate_class_attribute_name(field.name)
                mapping_fieldname = field.name
                field_path = struct.structpath._add_path(attrname)
                if isinstance(field.type, ScalarType):
                    if field.type.is_none():
                        # mypy does not allow to save the return of a function returning None
                        struct_code_buffer.write(
                            f'{field_indent}{attrname}=None if check_for_None(data.pop("{mapping_fieldname}", None)) else None,\n'
                        )
                    else:
                        function_name = f"extract_{field.type.to_python_varname(self._settings, field_path)}"
                        struct_code_buffer.write(
                            f'{field_indent}{attrname}={function_name}(data.pop("{mapping_fieldname}")),\n'
                        )
                elif isinstance(field.type, Literal):
                    function_name = generate_literal_extract_def(field.type, field_path)
                    value_extract = f'data.pop("{mapping_fieldname}")'
                    if field.type.has_nullable():
                        value_extract = f'data.pop("{mapping_fieldname}", None)'
                    struct_code_buffer.write(f"{field_indent}{attrname}={function_name}({value_extract}),\n")
                elif isinstance(field.type, Union):
                    function_name = generate_union_extract_def(field.type, field_path)
                    value_extract = f'data.pop("{mapping_fieldname}")'
                    if field.type.has_nullable():
                        value_extract = f'data.pop("{mapping_fieldname}", None)'
                    struct_code_buffer.write(
                        f"{field_indent}{attrname}={function_name}({value_extract}, check_for_remaining_keys),\n"
                    )
                elif isinstance(field.type, Structure):
                    function_name = generate_structure_extract_def(field.type, field_path)
                    struct_code_buffer.write(
                        f'{field_indent}{attrname}={function_name}(data.pop("{mapping_fieldname}"), check_for_remaining_keys),\n'
                    )
                elif isinstance(field.type, Sequence):
                    function_name = generate_sequence_extract_def(field.type, field_path)
                    struct_code_buffer.write(
                        f'{field_indent}{attrname}={function_name}(data.pop("{mapping_fieldname}"), check_for_remaining_keys),\n'
                    )
                else:
                    assert_never(field.type)
            struct_code_buffer.write("        )\n")
            struct_code_buffer.write("        if check_for_remaining_keys and len(data) > 0:\n")
            struct_code_buffer.write("            raise ValueError('remaining field', data)\n")
            struct_code_buffer.write("        return obj\n")
            struct_code_buffer.write("\n\n")
            struct_code_buffer.flush()
            struct_code_buffer.seek(0)
            code_buffer.write(struct_code_buffer.read())
        if generate_from_mapping_methods:
            extract_defs = dedent(
                """\
                def check_for_None(value: typing.Any) -> bool:
                    if value is None:
                        return True
                    raise ValueError(f"unepected type: {type(value).__name__}")

                def extract_bool(value: typing.Any) -> bool:
                    if not isinstance(value, bool):
                        raise ValueError(f"unepected type: {type(value).__name__}")
                    return value

                def extract_int(value: typing.Any) -> int:
                    if not isinstance(value, int):
                        raise ValueError(f"unepected type: {type(value).__name__}")
                    return value

                def extract_float(value: typing.Any) -> float:
                    if not isinstance(value, float):
                        raise ValueError(f"unepected type: {type(value).__name__}")
                    return value

                def extract_str(value: typing.Any) -> str:
                    if not isinstance(value, str):
                        raise ValueError(f"unepected type: {type(value).__name__}")
                    return value

                def extract_date(value: typing.Any) -> datetime.date:
                    conv_value = parse_date(value)
                    if conv_value is None:
                        raise ValueError(f"invalid date format: {value}")
                    return conv_value

                def extract_datetime(value: typing.Any) -> datetime.datetime:
                    conv_value = parse_datetime(value)
                    if conv_value is None:
                        raise ValueError(f"invalid datetime format: {value}")
                    return conv_value

                def extract_time(value: typing.Any) -> datetime.time:
                    conv_value = parse_time(value)
                    if conv_value is None:
                        raise ValueError(f"invalid time format: {value}")
                    return conv_value
                """
            )
            code_buffer.write(extract_defs)
            for f in (self._settings.parse_date, self._settings.parse_datetime, self._settings.parse_time):
                source = getsource(f)
                source = dedent(source)
                source = "\n".join(source.splitlines()[1:])
                code_buffer.write(source + "\n")
            for s in generate_extract_functions_strs:
                code_buffer.write("\n")
                code_buffer.write(s)


def parse_unstructured_data(
    root_name: str,
    data: Iterable[Mapping[str, RawValues]] | Mapping[str, RawValues],
    settings: type[Settings] = Settings,
) -> ParseResult:
    """Parse some unstructured data, either mapping like or sequence of mapping

    Args:
        root_name (str): name used for the root level object
        data (Iterable[Mapping[str, RawValues]] | Mapping[str, RawValues]): data you wish to parse
        settings (type[Settings], optional): settings used to customize the parsing and generation behavior. Defaults to Settings.

    Returns:
        ParseResult: the parsing result used to generate code
    """
    path = NamePath(root_name)
    if isinstance(data, Mapping):
        return ParseResult(settings, Structure.from_mapping(path, data, 0, settings))

    data_iterator = iter(data)
    try:
        first_mapping = next(data_iterator)
    except StopIteration:
        raise ValueError("no data found on iterator")
    dt = Structure.from_mapping(path, first_mapping, 0, settings)
    for mapping in data_iterator:
        new_dt = Structure.from_mapping(path, mapping, 0, settings)
        dt.merge(new_dt)
    return ParseResult(settings, dt)


async def parse_unstructured_data_async(
    root_name: str,
    data: AsyncIterable[Mapping[str, RawValues]],
    settings: type[Settings] = Settings,
) -> ParseResult:
    """Parse some unstructured data from an aynsc iterator

    Args:
        root_name (str): name used for the root level object
        data (AsyncIterable[Mapping[str, RawValues]]): data you wish to parse
        settings (type[Settings], optional): settings used to customize the parsing and generation behavior. Defaults to Settings.

    Returns:
        ParseResult: the parsing result used to generate code
    """
    path = NamePath(root_name)
    data_iterator = aiter(data)
    try:
        first_mapping = await anext(data_iterator)
    except StopIteration:
        raise ValueError("no data found on iterator")
    dt = Structure.from_mapping(path, first_mapping, 0, settings)
    async for mapping in data_iterator:
        new_dt = Structure.from_mapping(path, mapping, 0, settings)
        dt.merge(new_dt)
    return ParseResult(settings, dt)
