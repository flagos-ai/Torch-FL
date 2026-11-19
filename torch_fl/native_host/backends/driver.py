# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bind published FlagTree drivers to the wheel's prebuilt launch extensions."""

import importlib

from ..backend import BackendInterface


def reject_compilation(*args, **kwargs):
    raise RuntimeError("runtime C/C++ launcher compilation is forbidden")


def active_drivers():
    from triton.runtime import driver

    return (getattr(driver, "_active", None), getattr(driver, "_default", None))


def bind_instances(instances, utilities, create_launcher, current_stream=None):
    for instance in instances:
        if instance is None:
            continue
        instance.utils = utilities
        instance.launcher_cls = create_launcher
        if current_stream is not None:
            instance.get_current_stream = current_stream


def bind_driver(
    module,
    symbols,
    utilities,
    create_launcher,
    current_stream,
    *,
    utility_exports=(),
    compiler_hooks=("compile_module_from_src",),
):
    """Patch the utility/launcher factories and their existing and future drivers."""
    driver_name, utilities_name, launcher_name = symbols
    setattr(module, utilities_name, lambda: utilities)
    setattr(module, launcher_name, create_launcher)
    for name in utility_exports:
        setattr(module, name, getattr(utilities, name))
    for name in compiler_hooks:
        setattr(module, name, reject_compilation)

    module._flagos_native_stream = current_stream
    driver_type = getattr(module, driver_name)
    original_init = driver_type.__init__
    if not getattr(original_init, "_flagos_stream_provider", False):

        def initialize(instance, *args, **kwargs):
            original_init(instance, *args, **kwargs)
            instance.get_current_stream = module._flagos_native_stream

        initialize._flagos_stream_provider = True
        driver_type.__init__ = initialize

    bind_instances(active_drivers(), utilities, create_launcher, current_stream)


def install_driver(adapter: BackendInterface, package, launcher, create_launcher):
    config = adapter.config
    module = importlib.import_module(
        f"triton.backends.{config.driver_directory}.{config.driver_entry}"
    )
    utilities = importlib.import_module(f"{package}.{config.utility_module}")
    adapter.bind_driver(module, utilities, create_launcher, launcher.current_stream)
