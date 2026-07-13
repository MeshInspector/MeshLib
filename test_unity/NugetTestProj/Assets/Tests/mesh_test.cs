using NUnit.Framework;
using UnityEngine;

public class mesh_test
{
    // Builds a mesh and decimates it through the nuget package, so the native
    // library must load and compute, not just the managed wrapper.
    [Test]
    public void MeshLibNativeSmoke()
    {
        Debug.Log("MeshLibNativeSmoke begin");

        MR.Mesh sphere = MR.makeSphere(new MR.SphereParams(0.5f, 30000));
        Debug.Log($"makeSphere: {sphere.topology.getValidFaces().count()} valid faces");
        Assert.That(sphere.topology.getValidFaces().count() > 0);

        var settings = new MR.DecimateSettings();
        settings.maxError = 1e-3f;
        var result = MR.decimateMesh(sphere, settings);
        Debug.Log($"decimateMesh: facesDeleted={result.facesDeleted} vertsDeleted={result.vertsDeleted}");
        Assert.That(result.facesDeleted > 0);
        Assert.That(result.vertsDeleted > 0);

        Debug.Log("MeshLibNativeSmoke end");
    }
}
