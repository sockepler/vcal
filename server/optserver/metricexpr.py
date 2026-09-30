"""Preflight numerical metric expressions without requiring simulation data."""
import ast
import io
import math
import tokenize

SUFFIXES = {"T": 1e12, "G": 1e9, "M": 1e6, "meg": 1e6, "K": 1e3,
            "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12,
            "f": 1e-15, "a": 1e-18}
FUNCTIONS = {
    "abs", "min", "max", "float", "int", "round", "len", "sum", "log10", "log", "sqrt", "db",
    "V", "I", "avg", "rms", "vmax", "vmin", "at", "slice", "pp", "integ", "std",
    "cross", "rise_time", "fall_time", "delay", "slew_rate", "settle_time", "settled",
    "settling_error", "overshoot", "dc_gain", "dc_offset", "sample", "enob", "sndr_fft",
    "snr_fft", "thd_fft", "sfdr_fft", "fft_metrics", "adc_static", "adc_transitions", "decode_bits",
}
NUMPY_NAMES = {
    "abs", "absolute", "min", "max", "mean", "std", "sum", "sqrt", "square", "diff",
    "gradient", "interp", "trapezoid", "trapz", "where", "clip", "percentile", "median",
    "exp", "log", "log10", "sin", "cos", "tan", "arctan2", "real", "imag", "angle",
    "arange", "linspace", "array", "asarray", "column_stack", "ptp", "all", "any", "isfinite",
    "pi", "e", "fft", "fft.rfft", "fft.rfftfreq", "fft.fft", "fft.fftfreq",
}
ARRAY_METHODS = {"min", "max", "mean", "std", "sum", "astype"}
_ALLOWED = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp,
            ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.keyword,
            ast.Attribute, ast.Subscript, ast.Slice, ast.List, ast.Tuple, ast.Dict,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or, ast.BitAnd, ast.BitOr,
            ast.Invert, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def expand_literals(expr):
    """Expand SI suffixes, including .5n, without modifying quoted signal names."""
    tokens = list(tokenize.generate_tokens(io.StringIO(expr).readline))
    result, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == tokenize.NUMBER and index + 1 < len(tokens):
            following = tokens[index + 1]
            if (following.type == tokenize.NAME and following.string in SUFFIXES
                    and token.end == following.start):
                value = float(token.string) * SUFFIXES[following.string]
                if not math.isfinite(value):
                    raise ValueError("engineering literal is not finite")
                result.append((tokenize.NUMBER, repr(value)))
                index += 2
                continue
        result.append((token.type, token.string))
        index += 1
    return tokenize.untokenize(result)


def _numpy_path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == "np":
        return ".".join(reversed(parts))
    return None


def parse_expression(expr):
    """Return compiled code and literal m['name'] dependencies."""
    tree = ast.parse(expand_literals(expr), mode="eval")
    dependencies = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError("unsupported expression construct: " + type(node).__name__)
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, complex)):
            if isinstance(node.value, complex) or not math.isfinite(node.value):
                raise ValueError("metric constants must be finite real values")
        if isinstance(node, ast.Name) and node.id not in FUNCTIONS | {"np", "m", "t", "axis"}:
            raise ValueError("unknown metric symbol: " + node.id)
        if isinstance(node, ast.Attribute):
            path = _numpy_path(node)
            if path is not None:
                if path not in NUMPY_NAMES:
                    raise ValueError("unsupported numpy operation: np." + path)
            elif node.attr not in ARRAY_METHODS:
                raise ValueError("unsupported attribute: " + node.attr)
        if isinstance(node, ast.keyword) and node.arg is None:
            raise ValueError("keyword expansion is not supported in metrics")
        if isinstance(node, ast.Call) and not isinstance(node.func, (ast.Name, ast.Attribute)):
            raise ValueError("metric calls must use a named numerical function")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in FUNCTIONS:
            raise ValueError("not a metric function: " + node.func.id)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "m":
            key = node.slice
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ValueError("metric references must use m['literal_name']")
            dependencies.add(key.value)
    return compile(tree, "<metric>", "eval"), dependencies


def prepare_definitions(defs, analyses=None, default_analysis=None):
    """Normalize and dependency-sort expressions before starting jobs."""
    if not isinstance(defs, dict) or not defs:
        raise ValueError("metrics must be a nonempty mapping")
    available = tuple(analyses) if analyses is not None else ("tran", "dc")
    default = default_analysis or ("tran" if "tran" in available else "dc")
    entries, errors = {}, []
    for name, raw in defs.items():
        try:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("metric name must be a nonempty string")
            if isinstance(raw, str):
                expr, analysis = raw, default
            elif isinstance(raw, dict) and set(raw) <= {"expr", "analysis"}:
                expr, analysis = raw.get("expr"), raw.get("analysis", default)
            else:
                raise ValueError("definition must be an expression or {expr, analysis}")
            if not isinstance(expr, str) or not expr.strip():
                raise ValueError("expression must be a nonempty string")
            if analysis not in available:
                raise ValueError("analysis %r is not configured (available: %s)" % (analysis, ", ".join(available)))
            code, dependencies = parse_expression(expr)
            missing = dependencies - set(defs)
            if missing:
                raise ValueError("unknown referenced metric(s): " + ", ".join(sorted(missing)))
            entries[name] = {"expr": expr, "analysis": analysis, "code": code,
                             "dependencies": dependencies}
        except (ValueError, TypeError, SyntaxError, tokenize.TokenError) as exc:
            errors.append("metric %r: %s" % (name, exc))
    if errors:
        raise ValueError("Invalid metric definitions:\n- " + "\n- ".join(errors))
    ordered, pending = {}, dict(entries)
    while pending:
        ready = [name for name, entry in pending.items() if entry["dependencies"] <= set(ordered)]
        if not ready:
            raise ValueError("cyclic metric dependencies: " + ", ".join(pending))
        for name in ready:
            ordered[name] = pending.pop(name)
    return ordered


def validate_metric_definitions(defs, analyses=None, default_analysis=None):
    entries = prepare_definitions(defs, analyses, default_analysis)
    return {name: {"expr": item["expr"], "analysis": item["analysis"]}
            for name, item in entries.items()}
