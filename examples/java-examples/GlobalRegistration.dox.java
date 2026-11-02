import java.util.ArrayList;
import meshlib.AffineXf3f;
import meshlib.Box3f;
import meshlib.ICPProperties;
import meshlib.MeshOrPoints;
import meshlib.MeshOrPointsXf;
import meshlib.MultiwayICP;
import meshlib.MultiwayICPSamplingParameters;
import meshlib.ObjId;
import meshlib.PointCloud;
import meshlib.Vector_MR_AffineXf3f_MR_ObjId;
import meshlib.Vector_MR_MeshOrPointsXf_MR_ObjId;
import meshlib.VertCoords;
import meshlib.VertId;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class GlobalRegistration
{
    static void printStats(MultiwayICP icp)
    {
        System.out.println("Samples: " + icp.getNumSamples());
        System.out.println("Active point pairs: " + icp.getNumActivePairs());
        if (icp.getNumActivePairs() > 0)
        {
            float p2ptMetric = icp.getMeanSqDistToPoint(null);
            float p2plMetric = icp.getMeanSqDistToPlane(null);
            System.out.println("RMS point-to-point distance: " + p2ptMetric + " ± " + icp.getMeanSqDistToPoint((double)p2ptMetric));
            System.out.println("RMS point-to-plane distance: " + p2plMetric + " ± " + icp.getMeanSqDistToPlane((double)p2plMetric));
        }
    }

    public static void main(String[] args)
    {
        if (args.length < 3)
        {
            System.err.println("Usage: GlobalRegistration INPUT1 INPUT2 [INPUTS...] OUTPUT");
            System.exit(1);
        }

        // the global registration can be applied to meshes and point clouds
        // to simplify the sample app, we will work with point clouds only
        ArrayList<PointCloud> inputs = new ArrayList<>();
        // as ICP and MultiwayICP classes accept both meshes and point clouds,
        // the input data must be converted to special wrapper objects
        // NB: the wrapper objects hold *references* to the source data, NOT their copies
        Vector_MR_MeshOrPointsXf_MR_ObjId objects = new Vector_MR_MeshOrPointsXf_MR_ObjId();
        Box3f maxBBox = new Box3f();
        for (int i = 0; i < args.length - 1; i++)
        {
            // An error is thrown as `UnexpectedResultException`.
            PointCloud pointCloud = meshlib.pointsload.Functions.fromAnySupportedFormat(args[i], null);

            Box3f bbox = pointCloud.computeBoundingBox(null);
            if (!maxBBox.valid() || bbox.volume() > maxBBox.volume())
                maxBBox = bbox;

            inputs.add(pointCloud);
            // you may also set an affine transformation for each input as a second argument
            objects.push_back_const_MR_MeshOrPointsXf_ref(new MeshOrPointsXf(new MeshOrPoints(pointCloud), new AffineXf3f()));
        }

        // you can set various parameters for the global registration; see the documentation for more info
        MultiwayICPSamplingParameters samplingParams = new MultiwayICPSamplingParameters();
        // set sampling voxel size
        samplingParams.setSamplingVoxelSize(maxBBox.diagonal() * 0.03f);
        MultiwayICP icp = new MultiwayICP(objects, samplingParams);

        icp.setParams(new ICPProperties());

        // gather statistics
        icp.updateAllPointPairs(null);
        printStats(icp);

        System.out.println("Calculating transformations...");
        Vector_MR_AffineXf3f_MR_ObjId xfs = icp.calculateTransformations(null);
        printStats(icp);

        PointCloud output = new PointCloud();
        for (int i = 0; i < inputs.size(); i++)
        {
            VertCoords points = inputs.get(i).points();
            AffineXf3f xf = xfs.index(new ObjId(i));
            for (int v = 0; v < points.size(); v++)
                output.addPoint(xf.call(points.index(new VertId(v))));
        }

        // An error is thrown as `UnexpectedResultException`.
        meshlib.pointssave.Functions.toAnySupportedFormat(output, args[args.length - 1], null);
    }
}
