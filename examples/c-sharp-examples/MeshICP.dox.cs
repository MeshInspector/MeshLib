using System.Reflection;
using static MR.DotNet;

public class MeshICPExample
{
    public static void Run()
    {
        try
        {
            // Load meshes

            Mesh meshFloating = MeshLoad.FromAnySupportedFormat("meshA.stl");
            MeshOrPointsXf meshXfFloating = new MeshOrPointsXf(meshFloating, new AffineXf3f());
            MeshOrPointsXf meshXfFixed = new MeshOrPointsXf(MeshLoad.FromAnySupportedFormat("meshB.stl"), new AffineXf3f());

            // Prepare ICP parameters
            float diagonal = meshXfFixed.obj.BoundingBox.Diagonal();
            float icpSamplingVoxelSize = diagonal * 0.01f; // To sample points from object
            ICPProperties icpParams = new ICPProperties();
            icpParams.distThresholdSq = diagonal * diagonal * 0.01f; // Use points pairs with maximum distance specified
            icpParams.exitVal = diagonal * 0.003f; // Stop when distance reached

            // Calculate transformation
            ICP icp = new ICP(meshXfFloating, meshXfFixed, icpSamplingVoxelSize);
            icp.SetParams( icpParams );
            AffineXf3f xf = icp.CalculateTransformation();

            // Transform floating mesh
            meshFloating.Transform(xf);

            // Output information string
            Console.WriteLine("info {0}", icp.GetStatusInfo());

            // Save result
            MeshSave.ToAnySupportedFormat( meshFloating, "meshA_icp.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
