use meshlib::{FaceId, VertId};

fn make_sphere() -> meshlib::Owned<meshlib::Mesh>
{
    let mut params = meshlib::SphereParams::new();
    params.set_num_mesh_vertices(3000);
    params.set_radius(1.0);
    meshlib::make_sphere(&params)
}

#[test]
fn expand_shrink()
{
    let mesh = make_sphere();
    let top = mesh.topology();
    let mut region = meshlib::expand_mr_face_id(top, FaceId {id_: 0}, 3);
    assert_eq!(region.count(), 75);
    meshlib::expand_mr_face_bit_set(top, &mut region, Some(3));
    let num = region.count();
    assert!(num == 274 || // without FMA instruction (default settings for x86 or old compilers for ARM)
            num == 284); // with FMA instruction (modern compilers for ARM)
    meshlib::shrink_mr_face_bit_set(top, &mut region, Some(3));
    assert_eq!(region.count(), 75);
}

#[test]
fn expand_shrink_verts()
{
    let mesh = make_sphere();
    let top = mesh.topology();
    let mut region = meshlib::expand_mr_vert_id(top, VertId {id_: 0}, 3);
    assert_eq!(region.count(), 37);
    meshlib::expand_mr_vert_bit_set(top, &mut region, Some(3));
    assert!(region.count() > 37); // platform dependent results
    meshlib::shrink_mr_vert_bit_set(top, &mut region, Some(3));
    assert_eq!(region.count(), 37);
}
