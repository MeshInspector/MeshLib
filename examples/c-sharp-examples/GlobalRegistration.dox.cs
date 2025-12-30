using System.Reflection;

public class GlobalRegistrationExample
{
    static void PrintStats(MR.MultiwayICP icp)
    {
        ulong numActivePairs = icp.GetNumActivePairs();
        Console.WriteLine($"Number of samples: {icp.GetNumSamples()}");
        Console.WriteLine($"Number of active pairs: {numActivePairs}");

        if (numActivePairs > 0)
        {
            double p2ptMetric = icp.GetMeanSqDistToPoint();
            double p2ptInaccuracy = icp.GetMeanSqDistToPoint(p2ptMetric);
            Console.WriteLine($"RMS point-to-point distance: {p2ptMetric} ± {p2ptInaccuracy}");

            double p2plMetric = icp.GetMeanSqDistToPlane();
            double p2plInaccuracy = icp.GetMeanSqDistToPlane(p2ptMetric);
            Console.WriteLine($"RMS point-to-plane distance: {p2plMetric} ± {p2plInaccuracy}");
        }
    }
    public static void Run(string[] args)
    {
        if (args.Length < 4)
        {
            Console.WriteLine("Usage: {0} GlobalRegistrationExample INPUT1 INPUT2 [INPUTS...] OUTPUT", Assembly.GetExecutingAssembly().GetName().Name);
            return;
        }

        try
        {
            int inputNum = args.Length - 2;
            List<MR.PointCloud> input_pointclouds = new();
            MR.Vector_MRMeshOrPointsXf_MRObjId inputs = new();
            MR.Box3f maxBBox = new();
            for (int i = 0; i < inputNum; ++i)
            {
                var pc = MR.PointsLoad.FromAnySupportedFormat(args[i + 1]);
                input_pointclouds.Add(pc); // Need this to prevent the point-cloud object from dying too early.
                MR.MeshOrPointsXf obj = new MR.MeshOrPointsXf(pc, new MR.AffineXf3f());
                inputs.PushBack(obj);
                maxBBox.Include(obj.obj.ComputeBoundingBox());
            }

            MR.MultiwayICPSamplingParameters samplingParams = new();
            samplingParams.samplingVoxelSize = maxBBox.Diagonal() * 0.03f;

            MR.MultiwayICP icp = new(inputs, samplingParams);
            MR.ICPProperties iCPProperties = new();
            icp.SetParams(iCPProperties);
            icp.UpdateAllPointPairs();
            PrintStats(icp);

            Console.WriteLine("Calculating transformations...");
            MR.Vector_MRAffineXf3f_MRObjId xfs = icp.CalculateTransformations();
            PrintStats(icp);
            MR.PointCloud output = new();
            for (int i = 0; i < inputNum; ++i)
            {
                MR.ObjId id = new(i);
                var xf = xfs.Index(id);
                for (ulong j = 0; j < inputs.Index(id).obj.Points().Size(); j++)
                    output.AddPoint(xf.Call(inputs.Index(id).obj.Points().Index(new MR.VertId(j))));
            }

            MR.PointsSave.ToAnySupportedFormat(output, args[args.Length - 1]);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
