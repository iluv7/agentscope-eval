"""Self-contained evaluation suites discovered by the shared service."""

from importlib import import_module
from importlib.util import find_spec
from pkgutil import iter_modules


def iter_routers():
    """Yield the API router of each installed suite that provides one."""
    for module in sorted(iter_modules(__path__), key=lambda item: item.name):
        if not module.ispkg:
            continue
        name = f"{__name__}.{module.name}.api"
        if find_spec(name) is not None:
            yield import_module(name).router
