use crate::common::make_torus;

#[test]
fn mesh_collide()
{
    let mesh_a = make_torus(1.1, 0.5, 8, 8);
    let mesh_b = make_torus(1.1, 0.2, 8, 8);
    let part_a = meshlib::MeshPart::construct(&mesh_a, None);
    let part_b = meshlib::MeshPart::construct(&mesh_b, None);
    assert!(meshlib::is_inside_mr_mesh_part(&part_b, &part_a, None));
    assert!(!meshlib::is_inside_mr_mesh_part(&part_a, &part_b, None));
}
