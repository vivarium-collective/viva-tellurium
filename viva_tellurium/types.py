"""Custom bigraph-schema types for Tellurium wrapper.

Currently uses built-in types (map[float], overwrite, etc.) only.
This hook exists for future custom type registrations — e.g.
species-aware map types or unit-bearing concentrations.
"""


def register_tellurium_types(core):
    """Register custom types used by Tellurium processes. No-op for now."""
    return core
