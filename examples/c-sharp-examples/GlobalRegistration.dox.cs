using System.Reflection;
using static MR.DotNet;

public class GlobalRegistrationExample
{
    static void PrintStats(MultiwayICP icp)
    {
        int numActivePairs = icp.GetNumActivePairs();
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
            List<MeshOrPointsXf> inputs = new List<MeshOrPointsXf>(inputNum);
            Box3f maxBBox = new Box3f();
            for (int i = 0; i < inputNum; ++i)
            {
                MeshOrPointsXf obj = new MeshOrPointsXf(PointCloud.FromAnySupportedFormat(args[i + 1]), new AffineXf3f());
                inputs.Add(obj);
                Box3f bbox = obj.obj.BoundingBox;
                if (!maxBBox.Valid() || bbox.Volume() > maxBBox.Volume())
                    maxBBox = bbox;
            }

            MultiwayICPSamplingParameters samplingParams = new MultiwayICPSamplingParameters();
            samplingParams.samplingVoxelSize = maxBBox.Diagonal() * 0.03f;

            MultiwayICP icp = new MultiwayICP(inputs, samplingParams);
            ICPProperties iCPProperties = new ICPProperties();
            icp.SetParams(iCPProperties);
            icp.UpdateAllPointPairs();
            PrintStats(icp);

            Console.WriteLine("Calculating transformations...");
            var xfs = icp.CalculateTransformations();
            PrintStats(icp);
            PointCloud output = new PointCloud();
            for (int i = 0; i < inputNum; ++i)
            {
                var xf = xfs[i];
                for (int j = 0; j < inputs[i].obj.Points.Count; j++)
                    output.AddPoint(xf.Apply(inputs[i].obj.Points[j]));
            }

            PointCloud.ToAnySupportedFormat(output, args[args.Length - 1]);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
