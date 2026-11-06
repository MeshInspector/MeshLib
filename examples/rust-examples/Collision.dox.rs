use meshlib::{AffineXf3f, MeshPart, Vector3f};

fn main()
{
    let mesh_a = meshlib::make_uv_sphere(None, None, None); // make mesh A
    let mut mesh_b = meshlib::make_uv_sphere(None, None, None); // make mesh B
    mesh_b.transform(&AffineXf3f::translation(&Vector3f {x: 0.1, y: 0.1, z: 0.1}), None); // shift mesh B for better demonstration

    let part_a = MeshPart::construct(&mesh_a, None);
    let part_b = MeshPart::construct(&mesh_b, None);

    let colliding_face_pairs = meshlib::find_colliding_triangles(&part_a, &part_b, None, None); // find each pair of colliding faces
    for i in 0..colliding_face_pairs.size()
    {
        let pair = colliding_face_pairs.at(i);
        println!("{} {}", pair.a_face().id_, pair.b_face().id_); // print each pair of colliding faces
    }

    let colliding_face_bit_sets = meshlib::find_colliding_triangle_bitsets(&part_a, &part_b, None); // find bitsets of colliding faces
    println!("{}", colliding_face_bit_sets.first().count()); // print number of colliding faces from mesh A
    println!("{}", colliding_face_bit_sets.second().count()); // print number of colliding faces from mesh B

    let is_colliding = !meshlib::find_colliding_triangles(&part_a, &part_b, None, Some(true)).empty(); // fast check if mesh A and mesh B collide
    println!("{is_colliding}");
}
