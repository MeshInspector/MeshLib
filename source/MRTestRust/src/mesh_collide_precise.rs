use crate::common::make_torus;
use meshlib::mr;
use meshlib::mr::{AffineXf3f, Matrix3f, Vector3f};

#[test]
fn mesh_collide_precise()
{
    let mesh_a = make_torus(1.1, 0.5, 8, 8);
    let mut mesh_b = make_torus(1.1, 0.5, 8, 8);
    let rot = Matrix3f::rotation_mr_vector3f(&Vector3f::plus_z(), &Vector3f {x: 0.1, y: 0.8, z: 0.2});
    mesh_b.transform(&AffineXf3f::linear(&rot), None);

    let part_a = mr::MeshPart::construct(&mesh_a, None);
    let part_b = mr::MeshPart::construct(&mesh_b, None);
    let conv = mr::get_vector_converters_3(&part_a, &part_b, None);
    let intersections = mr::find_colliding_edge_tris_precise_5(&part_a, &part_b, conv.to_int(), None, None);
    assert_eq!(intersections.size(), 152);

    let contours = mr::order_intersection_contours(mesh_a.topology(), mesh_b.topology(), &intersections);
    assert_eq!(contours.size(), 4);
    let sizes: Vec<usize> = (0..4).map(|i| contours.at(i).size()).collect();
    assert!(sizes[0] == 69 || sizes[0] == 71); // Apple Debug
    assert!(sizes[1] == 9 || sizes[1] == 7); // Apple Debug
    assert!(sizes[2] == 69 || sizes[2] == 71); // with or without the FMA instruction
    assert!(sizes[3] == 9 || sizes[3] == 7); // with or without the FMA instruction

    let mut contours_a = mr::std_vector_MR_OneMeshContour::new();
    let mut contours_b = mr::std_vector_MR_OneMeshContour::new();
    mr::get_one_mesh_intersection_contours(&mesh_a, &mesh_b, &contours, Some(&mut contours_a), Some(&mut contours_b), &conv, None, None, None);
    assert_eq!(contours_a.size(), 4);
    assert_eq!(contours_b.size(), 4);
    let pos_count: usize = (0..contours_a.size()).map(|i| contours_a.at(i).intersections().size()).sum();
    assert_eq!(pos_count, 156);
}
