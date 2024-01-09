# Dataclass Generation using Retro engineering

A python library without external dependencies to generate dataclasses from one or several mappings (like dictionaries)

Can be used to roughly generate classes for an unknown source of data (e.g. an undocummented api or file for instance)

## Insallation

```sh
pip install git+https://github.com/alexandreseris/dataclass_retro_gen.git
```

## Usage

```python
from dataclass_retro_gen import Dataclass

with open("some_file_to_save_generated_classes", "w", encoding="utf8") as fs:
    # you can (and should if you can) also use Dataclass.from_mappings to generate better results
    dataclass_ = Dataclass.from_mapping("your_class_name", {"some": "dictionnary"})
    dataclass_.generate_class_definition(
        buffer=fs,  # you can also use io.StringIO if you prefer
        use_from_import=True,
        use_typing_union=False,
    )
```

## Limitations

there is no way to deal with aliases on field name, so a simple warning is just printed when the need of alias come

## Dev setup

```sh
# install with editable for ezpz changes
pip install --editable ".[dev]"

# format code
black .
# linter check
ruff check
# type checker
mypy .

# run tests suits
pytest .
```
