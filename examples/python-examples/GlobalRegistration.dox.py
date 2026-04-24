import meshlib.mrmeshpy as mrmeshpy
import sys


def print_stats(icp):
    num_active_pairs = icp.getNumActivePairs()
    print(f"Samples: {icp.getNumSamples()}")
    print(f"Active point pairs: {num_active_pairs}")

    if num_active_pairs > 0:
        p2pt_metric = icp.getMeanSqDistToPoint()
        p2pt_inaccuracy = icp.getMeanSqDistToPoint(p2pt_metric)
        print(f"RMS point-to-point distance: {p2pt_metric} ± {p2pt_inaccuracy}")

        p2pl_metric = icp.getMeanSqDistToPlane()
        p2pl_inaccuracy = icp.getMeanSqDistToPlane(p2pl_metric)
        print(f"RMS point-to-plane distance: {p2pl_metric} ± {p2pl_inaccuracy}")


def main(argv):
    if len(argv) < 4:
        print(f"Usage for script: {argv[0]} INPUT1 INPUT2 [INPUTS...] OUTPUT")
        return
    file_args = argv[1:]

    # the global registration can be applied to meshes and point clouds
    # to simplify the sample app, we will work with point clouds only
    input_clouds_num = len(file_args) - 1
    inputs = []
    # as ICP and MultiwayICP classes accept both meshes and point clouds,
    # the input data must be converted to special wrapper objects
    # NB: the wrapper objects hold *references* to the source data, NOT their copies
    max_bbox = None
    for i in range(input_clouds_num):
        points = mrmeshpy.loadPoints(file_args[i])
        # you may also set an affine transformation for each input as a second argument
        inputs.append(mrmeshpy.MeshOrPointsXf(points, mrmeshpy.AffineXf3f()))

        bbox = points.getBoundingBox()
        if not max_bbox or bbox.volume() > max_bbox.volume():
            max_bbox = bbox

    # you can set various parameters for the global registration; see the documentation for more info
    sampling_params = mrmeshpy.MultiwayICPSamplingParameters()
    # set sampling voxel size
    sampling_params.samplingVoxelSize = max_bbox.diagonal() * 0.03

    icp = mrmeshpy.MultiwayICP(mrmeshpy.Vector_MeshOrPointsXf_ObjId(inputs), sampling_params)
    icp.setParams(mrmeshpy.ICPProperties())

    # gather statistics
    icp.updateAllPointPairs()
    print_stats(icp)

    print("Calculating transformations...")
    xfs = icp.calculateTransformations()
    print_stats(icp)

    output = mrmeshpy.PointCloud()
    for i in range(input_clouds_num):
        xf = xfs.vec_[i]
        for point in inputs[i].obj.points():
            output.addPoint(xf(point))

    mrmeshpy.PointsSave.toAnySupportedFormat(output, file_args[-1])


main(sys.argv)
