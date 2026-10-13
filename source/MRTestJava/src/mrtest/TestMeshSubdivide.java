package mrtest;

import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.SubdivideSettings;

public final class TestMeshSubdivide
{
    public static void meshSubdivide()
    {
        Mesh mesh = Common.makeTorus(1.1f, 0.5f, 16, 16);
        SubdivideSettings settings = new SubdivideSettings();
        settings.setMaxDeviationAfterFlip(0.5f);
        Functions.subdivideMesh(mesh, settings);
    }
}
