import copy
import io
from textwrap import dedent
from typing import Union
from dataclass_retro_gen.generate import Type, Dataclass


def type_to_str(type: Type, depth: int = 0):
    buffer: list[str] = []
    buffer.append(f"{'  ' * depth}{type.origin_type_name()}")
    for arg in type.args:
        buffer.append(type_to_str(arg, depth=depth + 1))
    return "\n".join(buffer)


def pretty_type(before_merge: Type, test: Type, merge: Type, expected: Type):
    buffer = [
        "***TEST FAILLED***",
        "--test--",
        type_to_str(before_merge),
        "\n--merge--",
        type_to_str(merge),
        "\n--expected--",
        type_to_str(expected),
        "\n--result--",
        type_to_str(test),
    ]
    return "\n".join(buffer)


def _test_merge(test: Type, merge: Type, expected: Type):
    before_merge = copy.deepcopy(test)
    test.merge(merge)
    assert test == expected, pretty_type(before_merge, test, merge, expected)


def test_convert_from_simple_type():
    a = Type.from_python_type(list[str | None])
    b = Type.from_python_type(list[str | None])
    c = Type.from_python_type(list[str])

    assert a == b
    assert a != c
    assert b != c


def test_convert_type_with_args_simple():
    _test_merge(Type(str), Type(str), Type(str))


def test_convert_type_with_args_simple_union():
    _test_merge(
        Type(str),
        Type(int),
        Type(
            Union,
            Type(str),
            Type(int),
        ),
    )


def test_convert_type_with_args_simple_list():
    _test_merge(
        Type(list, Type(str)),
        Type(list, Type(str)),
        Type(list, Type(str)),
    )


def test_convert_type_with_args_union_list():
    _test_merge(
        Type(list, Type(str)),
        Type(
            list,
            Type(
                Union,
                Type(str),
                Type(int),
            ),
        ),
        Type(
            list,
            Type(
                Union,
                Type(str),
                Type(int),
            ),
        ),
    )


def test_convert_type_with_args_union_list2():
    _test_merge(
        Type(
            list,
            Type(
                Union,
                Type(str),
                Type(int),
            ),
        ),
        Type(list, Type(str)),
        Type(
            list,
            Type(
                Union,
                Type(str),
                Type(int),
            ),
        ),
    )


def test_convert_type_with_args_list_res_union():
    _test_merge(
        Type(list, Type(str)),
        Type(list, Type(int)),
        Type(
            list,
            Type(
                Union,
                Type(str),
                Type(int),
            ),
        ),
    )


def test_convert_type_with_two_level_depth():
    _test_merge(
        Type(list, Type(str)),
        Type(list, Type(int)),
        Type(
            list,
            Type(
                Union,
                Type(str),
                Type(int),
            ),
        ),
    )


def _test_dataclass_code_generation(dataclass_: Dataclass, code: str):
    buffer = io.StringIO()
    dataclass_.generate_class_definition(buffer)
    code_generated = buffer.getvalue()
    assert code_generated == code, f"\nEXPECTED:\n{code}\nGOT:\n{code_generated}"


def test_generate_dataclass_simple():
    _test_dataclass_code_generation(
        Dataclass.from_mapping("test", {}),
        dedent(
            """\
            from dataclasses import dataclass

            @dataclass
            class Test:
                pass
            """
        ),
    )


def test_generate_dataclass_flat():
    _test_dataclass_code_generation(
        Dataclass.from_mapping("test", {"a": 123, "b": "somestring"}),
        dedent(
            """\
            from dataclasses import dataclass

            @dataclass
            class Test:
                a: int
                b: str
            """
        ),
    )


def test_generate_dataclass_1_level():
    _test_dataclass_code_generation(
        Dataclass.from_mapping("test", {"a": 123, "b": "somestring", "c": {"d": 1.2}}),
        dedent(
            """\
            from dataclasses import dataclass

            @dataclass
            class C:
                d: float

            @dataclass
            class Test:
                a: int
                b: str
                c: "C"
            """
        ),
    )


def test_generate_dataclass_1_level_with_list():
    _test_dataclass_code_generation(
        Dataclass.from_mapping("test", {"a": 123, "b": "somestring", "c": {"d": [1, 2, 3]}}),
        dedent(
            """\
            from dataclasses import dataclass
            from typing import Sequence

            @dataclass
            class C:
                d: Sequence[int]

            @dataclass
            class Test:
                a: int
                b: str
                c: "C"
            """
        ),
    )


def test_generate_dataclass_2_level_with_list():
    _test_dataclass_code_generation(
        Dataclass.from_mapping("test", {"a": 123, "b": "somestring", "c": {"d": [{"b": 123}]}}),
        dedent(
            """\
            from dataclasses import dataclass
            from typing import Sequence

            @dataclass
            class D:
                b: int

            @dataclass
            class C:
                d: Sequence["D"]

            @dataclass
            class Test:
                a: int
                b: str
                c: "C"
            """
        ),
    )


def test_generate_dataclass_from_simple_list():
    _test_dataclass_code_generation(
        Dataclass.from_mappings("Test", [{}, {}]),
        dedent(
            """\
            from dataclasses import dataclass

            @dataclass
            class Test:
                pass
            """
        ),
    )


def test_generate_dataclass_from_list_with_different_types():
    _test_dataclass_code_generation(
        Dataclass.from_mappings("Test", [{"a": 2}, {"a": "test", "b": 4}]),
        dedent(
            """\
            from dataclasses import dataclass

            @dataclass
            class Test:
                a: int | str
                b: int | None
            """
        ),
    )


def test_generate_dataclass_conflict_names():
    _test_dataclass_code_generation(
        Dataclass.from_mapping("a", {"a": {"a": 123}}),
        dedent(
            """\
            from dataclasses import dataclass

            @dataclass
            class AA:
                a: int

            @dataclass
            class A:
                a: "AA"
            """
        ),
    )


def test_generate_dataclass_aliases():
    _test_dataclass_code_generation(
        Dataclass.from_mapping("test", {"A": 123, "B": "somestring"}),
        dedent(
            """\
            from dataclasses import dataclass

            @dataclass
            class Test:
                a: int  # ALIAS = A
                b: str  # ALIAS = B
            """
        ),
    )
