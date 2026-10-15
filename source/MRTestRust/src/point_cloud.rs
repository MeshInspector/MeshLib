use crate::common::make_torus;
use meshlib::mr;

#[test]
fn triangulation()
{
    let mesh = make_torus(2.0, 1.0, 32, 32);
    let pc = mr::mesh_to_point_cloud(&mesh, Some(mr::VertNormalsMode::AreaWeighted), None);
    let restored = mr::triangulate_point_cloud(&pc, None, None);
    let restored = restored.value().expect("triangulatePointCloud failed");
    assert_eq!(restored.points().size(), 1024);
    let top = restored.topology();
    assert_eq!(top.get_valid_verts().count(), 1024);
    assert_eq!(top.find_hole_representive_edges(None).size(), 0);
}
