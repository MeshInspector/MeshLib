use crate::common::make_torus;
use meshlib::mr;

#[test]
fn mesh_collide()
{
    let mesh_a = make_torus(1.1, 0.5, 8, 8);
    let mesh_b = make_torus(1.1, 0.2, 8, 8);
    let part_a = mr::MeshPart::construct(&mesh_a, None);
    let part_b = mr::MeshPart::construct(&mesh_b, None);
    assert!(mr::is_inside_mr_mesh_part(&part_b, &part_a, None));
    assert!(!mr::is_inside_mr_mesh_part(&part_a, &part_b, None));
}
