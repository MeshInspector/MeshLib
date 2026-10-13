package mrtest;

import static mrtest.Common.PI_F;
import static mrtest.Common.check;

import meshlib.mr.DecimateResult;
import meshlib.mr.DecimateSettings;
import meshlib.mr.FaceBitSet;
import meshlib.mr.Functions;
import meshlib.mr.Mesh;

public final class TestMeshDecimate
{
    public static void meshDecimate()
    {
        Mesh mesh = Functions.makeCylinderAdvanced(0.5f, 0.5f, 0.0f, 20.0f / 180.0f * PI_F, 1.0f, 16);

        // Select all faces.
        FaceBitSet region = new FaceBitSet(mesh.topology().getValidFaces());
        FaceBitSet regionSaved = new FaceBitSet(region);

        // Setup and run the decimator.
        DecimateResult result;
        {
            DecimateSettings settings = new DecimateSettings();
            settings.setMaxError(0.001f);
            settings.setRegion(region);
            settings.setMaxTriangleAspectRatio(80.0f);
            result = Functions.decimateMesh(mesh, settings);
        }

        // Compare the regions and the deleted vertices and faces.
        check(!Functions.opEqual(regionSaved, region), "region changed");
        check(result.vertsDeleted() > 0, "verts deleted");
        check(result.facesDeleted() > 0, "faces deleted");
    }
}
