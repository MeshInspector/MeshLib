# What the C++ API of `mrviewerpy` (`source/MRPythonViewer`) can't spell: lists and tuples instead of the vector classes, `UiEntry.__repr__`.
# Runs in the namespace of `mrviewerpy` when it's imported (`mrbind_gen_python --extra-python`, see `generate_python.sh`).

import atexit as _atexit

# Close the viewer the way a closed window would when the interpreter exits with it still up.
_atexit.register(shutViewerAtExit)
del shutViewerAtExit

def uiListEntries(path):
    """uiListEntries(path: list[str]) -> list[meshlib.mrviewerpy.UiEntry]

List existing UI entries at the specified path.
Pass an empty list to see top-level groups.
Add group name to the end of the vector to see its contents.
When you find the button you need, pass it to `uiPressButton()`."""
    return [UiEntry(e) for e in _uiListEntries(path)]

def uiListAllEntries(rootPath):
    """uiListAllEntries(rootPath: list[str]) -> list[tuple[list[str], meshlib.mrviewerpy.UiEntry]]

Flat depth-first list of every UI entry in the subtree rooted at `rootPath`.
Pass an empty list for the whole tree.
Each element is a `(path, UiEntry)` tuple where `path[-1] == entry.name`."""
    return [(list(path), UiEntry(entry)) for path, entry in _uiListAllEntries(rootPath)]

def _uiEntryRepr(e):
    return f"<mrmesh.mrviewerpy.UiEntry '{e.name}' of type '{e.type.name}' status='{e.status}'>"

ConstUiEntry.__repr__ = _uiEntryRepr

def _optionalList(field):
    def get(self):
        v = field.__get__(self, type(self))
        return None if v is None else list(v)
    return property(get, doc=field.__doc__)

ConstUiValueString.allowed = _optionalList(ConstUiValueString.__dict__['allowed'])
