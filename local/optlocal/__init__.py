import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
# reuse the server evaluation engine and the client BO stack in-process
for _p in (os.path.join(_ROOT, "server"), os.path.join(_ROOT, "client")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
