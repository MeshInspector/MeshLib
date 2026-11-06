use crate::common::create_cube;

#[test]
fn mesh_normals()
{
    let cube = create_cube();
    assert_eq!(meshlib::compute_per_vert_normals(&cube).size(), 8);
    assert_eq!(meshlib::compute_per_face_normals(&cube).size(), 12);
}
