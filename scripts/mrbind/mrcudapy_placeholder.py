# `mrcudapy` where MeshLib is built without Cuda (macOS, Windows Arm): only `isCudaAvailable()`, which returns false.

def isCudaAvailable(driverVersion=None, runtimeVersion=None, computeMajor=None, computeMinor=None) -> bool:
    """
    Returns true if Cuda is present on this GPU.
    Since Cuda is not supported on this platform, this function always returns false.
    """
    return False
