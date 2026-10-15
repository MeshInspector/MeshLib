use meshlib::VertId;

#[test]
fn unite_close_vertices_4()
{
    let mut params = meshlib::SphereParams::new();
    params.set_radius(1.0);
    params.set_num_mesh_vertices(3000);
    let mut mesh = meshlib::make_sphere(&params);

    let mut vert_map = meshlib::VertMap::new();
    let united = meshlib::mesh_builder::unite_close_vertices_4(&mut mesh, 0.1, Some(false), Some(&mut vert_map));
    assert_eq!(united, 2230);
    assert_eq!(vert_map.index(VertId {id_: 1000}).id_, 42);
}
