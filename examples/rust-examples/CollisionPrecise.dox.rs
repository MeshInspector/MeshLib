use meshlib::{AffineXf3f, MeshPart, Vector3f};

fn main()
{
    let mesh_a = meshlib::make_uv_sphere(None, None, None); // make mesh A
    let mut mesh_b = meshlib::make_uv_sphere(None, None, None); // make mesh B
    mesh_b.transform(&AffineXf3f::translation(&Vector3f {x: 0.1, y: 0.1, z: 0.1}), None); // shift mesh B for better demonstration

    let part_a = MeshPart::construct(&mesh_a, None);
    let part_b = MeshPart::construct(&mesh_b, None);

    let converters = meshlib::get_vector_converters(&part_a, &part_b, None); // create converters to integer field (needed for absolute precision predicates)
    let colliding_face_edges = meshlib::find_colliding_edge_tris_precise(&part_a, &part_b, converters.to_int(), None, None); // find each intersecting edge/triangle pair
    // print pairs of edges triangles
    for i in 0..colliding_face_edges.size()
    {
        let vet = colliding_face_edges.index(i);
        if vet.is_edge_a_tri_b()
        {
            println!("edgeA: {}, triB: {}", vet.edge().id_, vet.tri().id_);
        }
        else
        {
            println!("triA: {}, edgeB: {}", vet.tri().id_, vet.edge().id_);
        }
    }
}
