from helper import *
import pytest

def test_text_mesh():
    params = mrmesh.SymbolMeshParams()
    params.text = "Hello World"
    params.pathToFontFile = mrmesh.Path( working_directory + "//Cousine-Regular.ttf" )
    textMesh = mrmesh.createSymbolsMesh( params )
    assert textMesh.volume() > 0

def test_aligned_text_mesh():
    sphere = mrmesh.makeSphere( mrmesh.SphereParams() )
    params = mrmesh.TextAlignParams()
    # find representatice point on mesh
    params.startPoint = mrmesh.findProjection( mrmesh.Vector3f(0,0,1), sphere ).mtp
    params.direction = mrmesh.Vector3f(1,0,0)
    params.text = "Hello World"
    params.fontHeight = 0.02
    params.pathToFontFile = mrmesh.Path( working_directory + "//Cousine-Regular.ttf" )
    # how deep to engrave
    params.surfaceOffset = 0.1
    alignedMesh = mrmesh.alignTextToMesh( sphere, params )
    gravedPart = mrmesh.boolean(sphere,alignedMesh,mrmesh.BooleanOperation.Intersection).mesh
    volume = gravedPart.volume() 
    assert (volume > 0)
    assert (volume < alignedMesh.volume() )