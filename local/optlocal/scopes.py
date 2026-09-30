"""Group circuit parameters by the sub-cell scopes they reference.

The functions in this module operate on the parameter dictionaries returned by
``optserver.circuit.Circuit.spec``.  They deliberately do not inspect or
rewrite a netlist: a parameter that has no known nominal value is left out of
the returned fixed-value mapping so that the simulator keeps its original
netlist value.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping


def _device_parts(device):
    """Return ``(scope, instance)`` for one device specification."""
    if not isinstance(device, str) or not device:
        raise ValueError("device specification must be a nonempty string")
    if "/" not in device:
        return "top", device
    scope, instance = device.split("/", 1)
    if scope in ("", "top"):
        scope = "top"
    if not instance:
        raise ValueError("device specification must include an instance")
    return scope, instance


def _param_devices(param):
    if not isinstance(param, Mapping):
        raise ValueError("parameter must be a mapping")
    devices = param.get("devices")
    if not isinstance(devices, (list, tuple)) or not devices:
        raise ValueError("parameter devices must be a nonempty list")
    return devices


def parameter_scopes(param):
    """Return the sorted, unique scopes referenced by *param*.

    A device without a scope prefix and a device prefixed with ``top/`` both
    belong to the ``top`` scope.
    """
    scopes = {_device_parts(device)[0] for device in _param_devices(param)}
    return tuple(sorted(scopes))


def _normalise_params(params):
    if not isinstance(params, (list, tuple)):
        raise ValueError("params must be a list")
    entries = []
    by_name = {}
    for index, param in enumerate(params):
        if not isinstance(param, Mapping):
            raise ValueError("params[%d] must be a mapping" % index)
        name = param.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("params[%d] must have a nonempty name" % index)
        if name in by_name:
            raise ValueError("duplicate parameter: %s" % name)
        # Validate devices once here, while retaining the original mappings.
        scopes = parameter_scopes(param)
        entry = (param, name, scopes)
        entries.append(entry)
        by_name[name] = entry
    return entries, by_name


def scope_inventory(params):
    """Describe the parameters and device instances used by each scope."""
    entries, _ = _normalise_params(params)
    scopes = set()
    for _, _, param_scopes in entries:
        scopes.update(param_scopes)

    inventory = []
    for scope in sorted(scopes):
        parameters = []
        shared = []
        devices = set()
        for param, name, param_scopes in entries:
            if scope not in param_scopes:
                continue
            if len(param_scopes) == 1:
                parameters.append(name)
            else:
                shared.append(name)
            for device in _param_devices(param):
                device_scope, instance = _device_parts(device)
                if device_scope == scope:
                    devices.add(instance)
        inventory.append({
            "scope": scope,
            "parameters": sorted(set(parameters)),
            "shared": sorted(set(shared)),
            "devices": sorted(devices),
        })
    return inventory


def _mapping_copy(value, label):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("%s must be a mapping" % label)
    return dict(value)


def _bounds(param, name):
    try:
        lo = float(param["lo"])
        hi = float(param["hi"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("parameter %s has invalid bounds" % name) from exc
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError("parameter %s has invalid bounds" % name)
    return lo, hi


def _value(value, param, name, source):
    """Validate and normalize a fixed or initial value."""
    if isinstance(value, bool):
        raise ValueError("%s.%s must be a finite number" % (source, name))
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s.%s must be a finite number" % (source, name)) from exc
    if not math.isfinite(number):
        raise ValueError("%s.%s must be a finite number" % (source, name))
    lo, hi = _bounds(param, name)
    if not lo <= number <= hi:
        raise ValueError("%s.%s is outside parameter bounds" % (source, name))
    if param.get("integer", False) and not number.is_integer():
        raise ValueError("%s.%s must be an integer" % (source, name))
    return number


def select_scope(params, scope=None, *, fixed=None, initial_values=None):
    """Select parameters for one sub-cell scope and fix the rest.

    ``scope`` of ``None`` or ``""`` selects all scopes.  For a named scope a
    parameter is active only when every one of its devices belongs to that
    scope; parameters shared by scopes are therefore fixed.  Values supplied
    in ``initial_values`` take precedence over ``fixed`` and then ``nominal``
    for parameters that are not active.
    """
    entries, by_name = _normalise_params(params)
    fixed_input = _mapping_copy(fixed, "fixed")
    initial_input = _mapping_copy(initial_values, "initial_values")

    unknown_fixed = [name for name in fixed_input if name not in by_name]
    if unknown_fixed:
        raise ValueError("unknown parameter(s) in fixed: %s" %
                         ", ".join(str(name) for name in unknown_fixed))
    unknown_initial = [name for name in initial_input if name not in by_name]
    if unknown_initial:
        raise ValueError("unknown parameter(s) in initial_values: %s" %
                         ", ".join(str(name) for name in unknown_initial))

    all_scopes = {scope_name for _, _, param_scopes in entries
                  for scope_name in param_scopes}
    if scope in (None, ""):
        selected_scope = None
    else:
        if not isinstance(scope, str) or scope not in all_scopes:
            raise ValueError("unknown scope: %s" % scope)
        selected_scope = scope

    # Validate every supplied value before selecting parameters.  This keeps a
    # typo or an invalid value from being hidden merely because its parameter
    # is disabled in the selected view.
    fixed_values_input = {}
    for name, value in fixed_input.items():
        fixed_values_input[name] = _value(value, by_name[name][0], name, "fixed")
    initial_values_input = {}
    for name, value in initial_input.items():
        initial_values_input[name] = _value(value, by_name[name][0], name,
                                            "initial_values")

    active_params = []
    fixed_values = {}
    for param, name, param_scopes in entries:
        in_scope = selected_scope is None or set(param_scopes) == {selected_scope}
        enabled = param.get("enabled", True)
        is_active = in_scope and bool(enabled) and name not in fixed_values_input
        if is_active:
            active = copy.deepcopy(dict(param))
            if name in initial_values_input:
                active["nominal"] = initial_values_input[name]
            active_params.append(active)
            continue

        if name in initial_values_input:
            fixed_values[name] = initial_values_input[name]
        elif name in fixed_values_input:
            fixed_values[name] = fixed_values_input[name]
        elif param.get("nominal") is not None:
            fixed_values[name] = _value(param["nominal"], param, name,
                                        "nominal")
        # A None nominal deliberately means: leave the original netlist value.

    if not active_params:
        raise ValueError("at least one parameter must be active")
    return active_params, fixed_values


__all__ = ["parameter_scopes", "scope_inventory", "select_scope"]
