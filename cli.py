#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import inspect
import json
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union


NoneType = type(None)


class ScaffoldHelpFormatter(argparse.HelpFormatter):
    def __get_help(self, action) -> str:
        if inspect.isclass(action.type) and issubclass(action.type, Enum):
            help = "[%s]" % ",".join([str(e.value) for e in action.type])
            if action.required is False and action.default:
                help += " = %s" % str(action.default.value)
        else:
            help = action.type.__name__
            if action.required is False:
                help += " = %s" % str(action.default)
        return help

    def _get_default_metavar_for_optional(self, action) -> str:
        return self.__get_help(action)

    def _get_default_metavar_for_positional(self, action) -> str:
        return self.__get_help(action)


ArgComposerT = TypeVar("ArgComposerT", bound=NamedTuple)


class ArgParseArg(NamedTuple):
    name: str
    deserializer: Any
    serializer: Any
    default: Any
    value: Optional[str] = None


class ArgParseArgs(NamedTuple):
    required: List[ArgParseArg]
    optional: List[ArgParseArg]


T = TypeVar("T")


def _extract_optional_sub(type: Type[Optional[T]]) -> Tuple[T, bool]:
    required = True
    if callable(getattr(type, "_subs_tree", None)):
        subs = type._subs_tree()  # pyre-ignore
        if (subs[0] == Union) and (NoneType in subs):
            required = False
            type = subs[1] if subs[2] == NoneType else subs[2]
    return type, required


def arguments_from_named_tuple(
    named_tuple_ref: Type[ArgComposerT],
    named_tuple_instance: Optional[ArgComposerT] = None,
    prefix: str = "",
    optional_subtree: bool = False,
) -> ArgParseArgs:
    """
    Take a NamedTuple subclass and walk its members recursively to convert it into
    a set of descriptions of argparse command line arguments. If an instance of
    the NamedTuple is passed in, also associate values with each description to
    make it easy to reconstruct the string command line command that would reproduce
    the current run.
    """
    try:
        required_args: List[ArgParseArg] = []
        optional_args: List[ArgParseArg] = []
        # Once again python typehints are broken.. NamedTuples do not satisfy the
        # NamedTuple type var bound
        for field, type in named_tuple_ref.__annotations__.items():  # noqa T484
            field_val = (
                named_tuple_instance.__getattribute__(field)
                if named_tuple_instance
                else None
            )

            type, required = _extract_optional_sub(type)

            default = None
            if field in named_tuple_ref._field_defaults:  # noqa T484
                default = named_tuple_ref._field_defaults[field]  # noqa T484
                required = False

            if (
                inspect.isclass(type)
                and issubclass(type, tuple)
                and hasattr(type, "__annotations__")
            ):
                # Currently issubclass(NamedTupleSubclass, NamedTuple) == false
                child_arguments = arguments_from_named_tuple(  # noqa T484
                    type,
                    named_tuple_instance=field_val,
                    prefix=prefix + field + ".",
                    optional_subtree=(required == False),
                )
                optional_args += child_arguments.optional
                if required:
                    required_args += child_arguments.required
                else:
                    optional_args += child_arguments.required

                continue

            deserializer: Any = type
            serializer: Any = str
            # use 0/1 to indicate false/true for boolean type
            if type is bool:

                def bool_deserializer(x: str) -> bool:
                    return int(x) != 0

                def bool_serializer(x: bool) -> str:
                    return str(int(x))

                deserializer = bool_deserializer
                serializer = bool_serializer
            elif issubclass(type, (dict, list)):
                deserializer = json.loads
                serializer = json.dumps
            elif issubclass(type, tuple):

                def tuple_deserializer(x):
                    return tuple(json.loads(x))

                deserializer = tuple_deserializer
                serializer = json.dumps
            elif issubclass(type, Enum):

                def enum_value(x):
                    return x.value

                serializer = enum_value

            arg = ArgParseArg(
                name="--" + prefix + field,
                deserializer=deserializer,
                serializer=serializer,
                default=None if optional_subtree else default,
                value=field_val,
            )
            if required:
                required_args.append(arg)
            else:
                optional_args.append(arg)
        return ArgParseArgs(required=required_args, optional=optional_args)
    except AttributeError:
        raise RuntimeError("arguments_from_named_tuple requires a NamedTuple ref")


def compose_type_arguments(
    named_tuple_ref: Type[ArgComposerT],
    optionalParser: argparse._ArgumentGroup,
    requiredParser: argparse._ArgumentGroup,
) -> None:
    """
    Take a NamedTuple subclass and walk its members recursively to convert it into
    a set of argparse command line flags. Members with defaults become optional
    args while members without become required args.
    """
    arguments = arguments_from_named_tuple(named_tuple_ref)
    for arg in arguments.required:
        requiredParser.add_argument(
            arg.name, type=arg.deserializer, required=True, default=arg.default
        )
    for arg in arguments.optional:
        requiredParser.add_argument(
            arg.name, type=arg.deserializer, required=False, default=arg.default
        )


def compose_type_results(
    named_tuple_ref: Type[ArgComposerT],
    args: Dict[str, Any],
    prefix: str = "",
    required: bool = True,
) -> Optional[ArgComposerT]:
    """
    Take an untyped dictionary of arguments returned by argparse and convert
    this into a NamedTuple subclass, recursively.
    """
    # Once again python typehints are broken.. NamedTuples do not satisfy the
    # NamedTuple type var bound
    try:
        kwargs: Dict[str, Any] = {}
        for field, type in named_tuple_ref.__annotations__.items():  # noqa T484
            type, required_field = _extract_optional_sub(type)
            if (
                inspect.isclass(type)
                and issubclass(type, tuple)
                and hasattr(type, "__annotations__")
            ):
                # Currently issubclass(NamedTupleSubclass, NamedTuple) == false
                kwargs[field] = compose_type_results(  # noqa T484
                    type, args, prefix=prefix + field + ".", required=required_field
                )
                continue
            if prefix + field in args:
                kwargs[field] = args[prefix + field]
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not filtered_kwargs and not required:
            return None
        return named_tuple_ref(**filtered_kwargs)
    except AttributeError:
        raise RuntimeError("compose_type_results requires a NamedTuple ref")


TSolverOpts = TypeVar("TSolverOpts", bound=NamedTuple)
TProblemOpts = TypeVar("TProblemOpts", bound=NamedTuple)


def get_command_line_args(
    solver_ref: Type[TSolverOpts],
    problem_ref: Type[TProblemOpts],
    args_list: Optional[List[str]] = None,
) -> Tuple[TSolverOpts, TProblemOpts]:
    parser = argparse.ArgumentParser(formatter_class=ScaffoldHelpFormatter)
    requiredGroup = parser.add_argument_group("required arguments")
    optionalGroup = parser.add_argument_group("optional arguments")

    compose_type_arguments(solver_ref, optionalGroup, requiredGroup)
    compose_type_arguments(problem_ref, optionalGroup, requiredGroup)

    args = vars(parser.parse_args(args_list))

    solver_opts = compose_type_results(solver_ref, args)
    problem_opts = compose_type_results(problem_ref, args)
    assert solver_opts is not None
    assert problem_opts is not None
    return (solver_opts, problem_opts)


TNamedTuple = TypeVar("TNamedTuple", bound=NamedTuple)


def get_repro_args(*all_opts: TNamedTuple) -> str:
    argstr = ""
    for opts in all_opts:
        args = arguments_from_named_tuple(opts.__class__, opts)
        for arg in args.required + args.optional:
            if arg.value is not None:
                val = arg.serializer(arg.value)
                if " " in val:
                    if '"' in val:
                        val = f"'{val}'"
                    elif '"' in val and "'" in val:
                        raise RuntimeError(
                            "Unsure how to handle parameters with both single and double quotes: %s"
                            % val
                        )
                    else:
                        val = f'"{val}"'

                argstr += " " + arg.name + " " + val
    return argstr
