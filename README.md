# Dataclass Generation using Retro engineering

A python library to generate dataclasses from one or several mappings (like dictionaries)

Can be used to roughly generate classes for an unknown source of data (e.g. an undocummented api or file for instance)

## Insallation

```sh
pip install git+https://github.com/alexandreseris/dataclass_retro_gen.git
```

## Usage

```python
from dataclass_retro_gen import parse_unstructured_data

parse_result = parse_unstructured_data("root_key_name_of_your_choice", {"some": "dictionnary"})
with open("some_file_to_save_generated_classes", "w", encoding="utf8") as fs:
    parse_result.write_definitions(
        code_buffer=fs,  # you can also use io.StringIO if you prefer
        generate_from_dict_methods=True, # use this flag to generate a simple from_dict class method
    )
```

The way data is parsed and generated can be customize using `Settings` which can be passed to the `parse_unstructured_data` function. Please check the docstring for more details.

`Settings` is a protocol, you can either define your own protocol or inherit from `Settings` to change specific behavior.

You can also use `parse_unstructured_data_async` if you need to parse data from an ayns iterator

## Dev setup

```sh
# install with editable for ezpz changes
**pip install --editable ".[dev]"**

# format code
black .
# linter check
ruff check
# type checker
mypy .
```
