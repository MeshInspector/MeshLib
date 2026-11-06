use meshlib::MeshPart;

fn main() -> Result<(), String>
{
    let mesh = meshlib::make_torus_with_self_intersections(None, None, None, None, None); // make torus with self-intersections
    let part = MeshPart::construct(&mesh, None);

    let self_colliding_pairs = meshlib::find_self_colliding_triangles(&part, None, None, None).into_result()?; // find self-intersecting faces pairs
    for i in 0..self_colliding_pairs.size()
    {
        let pair = self_colliding_pairs.at(i);
        println!("{} {}", pair.a_face().id_, pair.b_face().id_); // print each pair
    }

    let self_colliding_bit_set = meshlib::find_self_colliding_triangles_bs(&part, None, None, None).into_result()?; // find bitset of self-intersecting faces
    println!("{}", self_colliding_bit_set.count()); // print number of self-intersecting faces

    let is_self_colliding = meshlib::find_self_colliding_triangles_out_pairs(&part, None, None, None, None).into_result()?; // fast check if mesh has self-intersections
    println!("{is_self_colliding}");

    Ok(())
}
