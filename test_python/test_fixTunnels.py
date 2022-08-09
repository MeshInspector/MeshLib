from helper import *
import pytest

def test_fixTunnels():
    torus = mrmesh.make_torus(2, 1, 10, 10, None)

    tunnelFaces = mrmesh.get_tunnel_faces(mrmesh.MeshPart(torus), 100500)

    # one circle with 2-faces width
    assert (tunnelFaces.count() == 20)
