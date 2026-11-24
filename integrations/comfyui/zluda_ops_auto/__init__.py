import os
import sys

root = os.environ.get("ZLUDA_ROOT", r"G:\zluda")
if root not in sys.path:
    sys.path.insert(0, root)
import zluda_ops.auto

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
