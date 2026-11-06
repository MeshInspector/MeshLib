fn main()
{
    // create some mesh
    let mesh = meshlib::make_cube(None, None);

    // all vertices of valid triangles
    let triangles = mesh.topology().get_all_tri_verts();

    // all point coordinates
    let points = mesh.points();
    // triangle vertices as triples of ints (pointing to elements in points vector)
    let vertex_triples = triangles.data() as *const i32;

    // TODO: export in your format
    let _ = points;
    let _ = vertex_triples;
}
