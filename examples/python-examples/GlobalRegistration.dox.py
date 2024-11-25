import meshlib.mrmeshpy as mr
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

def main():
    # if len(argv) < 4:
    #     print(f"Usage: {argv[0]} INPUT1 INPUT2 [INPUTS...] OUTPUT")
    #     return
    inp_files = ['']
    inp_files.append("cloud0.ply")
    inp_files.append("cloud1.ply")
    inp_files.append("cloud2.ply")
    inp_files.append("out.ply")

    input_num = len(inp_files) - 2
    inputs = []
    max_bbox = None

    for i in range(input_num):
        points = mr.loadPoints(inp_files[i + 1])
        xf = mr.AffineXf3f()
        obj = mr.MeshOrPointsXf(points, xf)
        inputs.append(obj)

        bbox = points.getBoundingBox()
        if not max_bbox or bbox.volume() > max_bbox.volume():
            max_bbox = bbox

    sampling_params = mr.MultiwayICPSamplingParameters()
    sampling_params.samplingVoxelSize = max_bbox.diagonal() * 0.03

    inputs_obj = mr.Vector_MeshOrPointsXf_ObjId(inputs)

    icp = mr.MultiwayICP(inputs_obj, sampling_params)
    icp_properties = mr.ICPProperties()
    icp.setParams(icp_properties)

    icp.updateAllPointPairs() # crash here

    print("Calculating transformations...")
    xfs = icp.calculateTransformations()
    print_stats(icp)

    output = mr.PointCloud()
    for i in range(input_num):
        xf = xfs.vec_[i]
        for point in inputs[i].obj.points():
            transformed_point = xf(point)
            output.addPoint(transformed_point)

    mr.PointsSave.toAnySupportedFormat(output, inp_files[-1])

main()