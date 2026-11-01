use meshlib::VertId;

fn main() -> Result<(), String>
{
    let ref_mesh = meshlib::mesh_load::from_any_supported_format("mesh1.ctm", None).into_result()?;
    let mesh = meshlib::mesh_load::from_any_supported_format("mesh2.ctm", None).into_result()?;

    // get object of VertScalars - set of distances between points of target mesh and reference mesh
    let vert_distances = meshlib::find_signed_distances(&ref_mesh, &mesh, None, None);
    let distances: Vec<f32> = (0..vert_distances.size()).map(|i| *vert_distances.index(VertId {id_: i as i32})).collect();
    let min = distances.iter().copied().fold(f32::INFINITY, f32::min);
    let max = distances.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!("Distance between reference mesh and the closest point of target mesh is {min}");
    println!("Distance between reference mesh and the farthest point of target mesh is {max}");

    Ok(())
}
