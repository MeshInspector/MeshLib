def _override_resources_dir():
    """
    override resources directory to the package's dir
    """
    import pathlib
    from . import mrmeshpy as mr

    mr.SystemPath.overrideDirectory(mr.SystemPath.Directory.Resources, pathlib.Path(__file__).parent.resolve())
    mr.SystemPath.overrideDirectory(mr.SystemPath.Directory.Fonts, pathlib.Path(__file__).parent.resolve())

_override_resources_dir()
del _override_resources_dir
