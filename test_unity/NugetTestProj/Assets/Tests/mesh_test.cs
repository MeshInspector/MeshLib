using NUnit.Framework;
using UnityEngine;

public class mesh_test
{
    // End-to-end smoke test of the MeshLib nuget package: builds a mesh and
    // decimates it, so the native library must load and compute, not just the
    // managed wrapper.
    [Test]
    public void MeshLibNativeSmoke()
    {
        MR.Mesh sphere = MR.makeSphere(new MR.SphereParams(0.5f, 30000));
        Assert.That(sphere.topology.getValidFaces().count() > 0);

        var settings = new MR.DecimateSettings();
        settings.maxError = 1e-3f;
        var result = MR.decimateMesh(sphere, settings);
        Assert.That(result.facesDeleted > 0);
        Assert.That(result.vertsDeleted > 0);

        Debug.Log("MeshTestPassed");
    }
}
