def _override_resources_dir():
    """
    override resources directory to the package's dir
    """
    import pathlib
    from . import mrviewerpy as mv

    mv.SystemPath.overrideDirectory(mv.SystemPath.Directory.Resources, pathlib.Path(__file__).parent.resolve())

_override_resources_dir()
del _override_resources_dir
