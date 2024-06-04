import pytest
from helper import *


def test_position_verts_smooth():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 12, None)

    params = mrmesh.LaplacianEdgeWeightsParam.Cotan
    verts = torus.topology.getValidVerts()
    verts.set(mrmesh.VertId(0), False)
    mrmesh.positionVertsSmoothly(torus, verts, params)

    p = torus.points.vec[0]
    for i in torus.points.vec:
        assert i.x * i.x + i.y * i.y + i.z * i.z == p.x * p.x + p.y * p.y + p.z * p.z
        
def test_position_verts_smooth_sharpbd():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 12, None)

    verts = torus.topology.getValidVerts()
    verts.set(mrmesh.VertId(0), False)
    mrmesh.positionVertsSmoothlySharpBd(torus, verts)

    p = torus.points.vec[0]
    for i in torus.points.vec:
        assert i.x * i.x + i.y * i.y + i.z * i.z == p.x * p.x + p.y * p.y + p.z * p.z

def test_inflate_verts_smooth():
    """
    Succession test that the function is exposed correctly
    """
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    
    smooth_region = mrmesh.VertBitSet()
    smooth_region.resize( 6, True )
    
    inlate_settings = mrmesh.InflateSettings()
    inlate_settings.pressure = -0.1
    
    mrmesh.inflate(torus, smooth_region, inlate_settings)