using System.Reflection;

public class GlobalRegistrationExample
{
    static void PrintStats(MR.MultiwayICP icp)
    {
        ulong numActivePairs = icp.getNumActivePairs();
        Console.WriteLine($"Number of samples: {icp.getNumSamples()}");
        Console.WriteLine($"Number of active pairs: {numActivePairs}");

        if (numActivePairs > 0)
        {
            double p2ptMetric = icp.getMeanSqDistToPoint();
            double p2ptInaccuracy = icp.getMeanSqDistToPoint(p2ptMetric);
            Console.WriteLine($"RMS point-to-point distance: {p2ptMetric} ± {p2ptInaccuracy}");

            double p2plMetric = icp.getMeanSqDistToPlane();
            double p2plInaccuracy = icp.getMeanSqDistToPlane(p2ptMetric);
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
            MR.Vector_MRMeshOrPointsXf_MRObjId inputs = new();
            MR.Box3f maxBBox = new();
            for (int i = 0; i < inputNum; ++i)
            {
                var pc = MR.PointsLoad.fromAnySupportedFormat(args[i + 1]);
                System.Runtime.InteropServices.GCHandle.Alloc(pc);
                MR.MeshOrPointsXf obj = new MR.MeshOrPointsXf(pc, new MR.AffineXf3f());
                inputs.pushBack(obj);
                maxBBox.include(obj.obj.computeBoundingBox());
            }

            MR.MultiwayICPSamplingParameters samplingParams = new();
            samplingParams.samplingVoxelSize = maxBBox.diagonal() * 0.03f;

            MR.MultiwayICP icp = new(inputs, samplingParams);
            MR.ICPProperties iCPProperties = new();
            icp.setParams(iCPProperties);
            icp.updateAllPointPairs();
            PrintStats(icp);

            Console.WriteLine("Calculating transformations...");
            MR.Vector_MRAffineXf3f_MRObjId xfs = icp.calculateTransformations();
            PrintStats(icp);
            MR.PointCloud output = new();
            for (int i = 0; i < inputNum; ++i)
            {
                MR.ObjId id = new(i);
                var xf = xfs[id];
                for (ulong j = 0; j < inputs[id].obj.points().size(); j++)
                    output.addPoint(xf.call(inputs[id].obj.points()[new MR.VertId(j)]));
            }

            MR.PointsSave.toAnySupportedFormat(output, args[args.Length - 1]);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
