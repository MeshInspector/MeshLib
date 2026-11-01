public static class GlobalRegistrationExample
{
    static void PrintStats(MR.MultiwayICP icp)
    {
        Console.WriteLine("Samples: {0}", icp.getNumSamples());
        Console.WriteLine("Active point pairs: {0}", icp.getNumActivePairs());
        if (icp.getNumActivePairs() > 0)
        {
            float p2ptMetric = icp.getMeanSqDistToPoint();
            float p2plMetric = icp.getMeanSqDistToPlane();
            Console.WriteLine("RMS point-to-point distance: {0} ± {1}", p2ptMetric, icp.getMeanSqDistToPoint(p2ptMetric));
            Console.WriteLine("RMS point-to-plane distance: {0} ± {1}", p2plMetric, icp.getMeanSqDistToPlane(p2plMetric));
        }
    }

    public static void Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("Usage: GlobalRegistration INPUT1 INPUT2 [INPUTS...] OUTPUT");
            Environment.Exit(1);
        }

        // the global registration can be applied to meshes and point clouds
        // to simplify the sample app, we will work with point clouds only
        List<MR.PointCloud> inputs = new();
        // as ICP and MultiwayICP classes accept both meshes and point clouds,
        // the input data must be converted to special wrapper objects
        // NB: the wrapper objects hold *references* to the source data, NOT their copies
        MR.Vector_MRMeshOrPointsXf_MRObjId objects = new();
        MR.Box3f maxBBox = new();
        for (int i = 0; i < args.Length - 1; ++i)
        {
            var pointCloud = MR.PointsLoad.fromAnySupportedFormat(args[i]);

            var bbox = pointCloud.computeBoundingBox();
            if (!maxBBox.valid() || bbox.volume() > maxBBox.volume())
                maxBBox = bbox;

            inputs.Add(pointCloud);
            // you may also set an affine transformation for each input as a second argument
            objects.pushBack(new MR.MeshOrPointsXf(pointCloud, new MR.AffineXf3f()));
        }

        // you can set various parameters for the global registration; see the documentation for more info
        MR.MultiwayICPSamplingParameters samplingParams = new();
        // set sampling voxel size
        samplingParams.samplingVoxelSize = maxBBox.diagonal() * 0.03f;
        MR.MultiwayICP icp = new(objects, samplingParams);

        icp.setParams(new MR.ICPProperties());

        // gather statistics
        icp.updateAllPointPairs();
        PrintStats(icp);

        Console.WriteLine("Calculating transformations...");
        var xfs = icp.calculateTransformations();
        PrintStats(icp);

        MR.PointCloud output = new();
        for (int i = 0; i < inputs.Count; ++i)
        {
            var points = inputs[i].points;
            var xf = xfs[new MR.ObjId(i)];
            for (int v = 0; v < (int)points.size(); ++v)
                output.addPoint(xf.call(points[new MR.VertId(v)]));
        }

        MR.PointsSave.toAnySupportedFormat(output, args[args.Length - 1]);
    }
}
