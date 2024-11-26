import os

import meshlib.mrmeshpy as mrmeshpy
import sys


def print_stats(icp):
    num_active_pairs = icp.getNumActivePairs()
    print(f"Number of samples: {icp.getNumSamples()}")
    print(f"Number of active pairs: {num_active_pairs}")

    if num_active_pairs > 0:
        p2pt_metric = icp.getMeanSqDistToPoint()
        p2pt_inaccuracy = icp.getMeanSqDistToPoint(value=p2pt_metric)
        print(f"RMS point-to-point distance: {p2pt_metric} ± {p2pt_inaccuracy}")

        p2pl_metric = icp.getMeanSqDistToPlane()
        p2pl_inaccuracy = icp.getMeanSqDistToPlane(p2pt_metric)
        print(f"RMS point-to-plane distance: {p2pl_metric} ± {p2pl_inaccuracy}")

def main(argv):
    if len(argv) < 4:
        print(f"Usage for script: {argv[0]} INPUT1 INPUT2 [INPUTS...] OUTPUT")
        return
    file_args = argv[1:]

    # Loading inputs and finding max bounding box
    input_clouds_num = len(file_args) - 1
    inputs = []
    max_bbox = None
    for i in range(input_clouds_num):
        points = mrmeshpy.loadPoints(file_args[i])
        transform = mrmeshpy.AffineXf3f()
        points_with_transform = mrmeshpy.MeshOrPointsXf(points, transform)
        inputs.append(points_with_transform)

        bbox = points.getBoundingBox()
        if not max_bbox or bbox.volume() > max_bbox.volume():
            max_bbox = bbox

    # ICP initialization
    sampling_params = mrmeshpy.MultiwayICPSamplingParameters()
    sampling_params.samplingVoxelSize = max_bbox.diagonal() * 0.03

    inputs_obj = mrmeshpy.Vector_MeshOrPointsXf_ObjId(inputs)

    icp = mrmeshpy.MultiwayICP(inputs_obj, sampling_params)
    icp_properties = mrmeshpy.ICPProperties()
    icp.setParams(icp_properties)

    icp.updateAllPointPairs()

    print("Calculating transformations...")
    xfs = icp.calculateTransformations()
    print_stats(icp)

    # Saving result
    output = mrmeshpy.PointCloud()
    for i in range(input_clouds_num):
        transform = xfs.vec_[i]
        for point in inputs[i].obj.points():
            transformed_point = transform(point)
            output.addPoint(transformed_point)

    mrmeshpy.PointsSave.toAnySupportedFormat(output, file_args[-1])

main(sys.argv)
