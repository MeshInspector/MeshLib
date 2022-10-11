from helper import *
import pytest


def test_fix_tunnels():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)

    tunnelFaces = mrmesh.detectTunnelFaces(mrmesh.MeshPart(torus), 100500)

    # one circle with 2-faces width
    assert (tunnelFaces.count() == 20)
