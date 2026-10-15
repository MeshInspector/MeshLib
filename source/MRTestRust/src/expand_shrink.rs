use meshlib::mr;
use meshlib::mr::{FaceId, VertId};

fn make_sphere() -> meshlib::Owned<mr::Mesh>
{
    let mut params = mr::SphereParams::new();
    params.set_num_mesh_vertices(3000);
    params.set_radius(1.0);
    mr::make_sphere(&params)
}

#[test]
fn expand_shrink()
{
    let mesh = make_sphere();
    let top = mesh.topology();
    let mut region = mr::expand_mr_face_id(top, FaceId {id_: 0}, 3);
    assert_eq!(region.count(), 75);
    mr::expand_mr_face_bit_set(top, &mut region, Some(3));
    let num = region.count();
    assert!(num == 274 || // without FMA instruction (default settings for x86 or old compilers for ARM)
            num == 284); // with FMA instruction (modern compilers for ARM)
    mr::shrink_mr_face_bit_set(top, &mut region, Some(3));
    assert_eq!(region.count(), 75);
}

#[test]
fn expand_shrink_verts()
{
    let mesh = make_sphere();
    let top = mesh.topology();
    let mut region = mr::expand_mr_vert_id(top, VertId {id_: 0}, 3);
    assert_eq!(region.count(), 37);
    mr::expand_mr_vert_bit_set(top, &mut region, Some(3));
    assert!(region.count() > 37); // platform dependent results
    mr::shrink_mr_vert_bit_set(top, &mut region, Some(3));
    assert_eq!(region.count(), 37);
}
