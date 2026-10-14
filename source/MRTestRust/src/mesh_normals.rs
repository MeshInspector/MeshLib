use crate::common::create_cube;
use meshlib::mr;

#[test]
fn mesh_normals()
{
    let cube = create_cube();
    assert_eq!(mr::compute_per_vert_normals(&cube).size(), 8);
    assert_eq!(mr::compute_per_face_normals_1(&cube).size(), 12);
}
