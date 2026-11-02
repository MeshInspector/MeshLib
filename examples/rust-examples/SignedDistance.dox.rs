use meshlib::MeshPart;

fn main() -> Result<(), String>
{
    let mesh1 = meshlib::mesh_load::from_any_supported_format("mesh1.ctm", None).into_result()?;
    let mesh2 = meshlib::mesh_load::from_any_supported_format("mesh2.ctm", None).into_result()?;

    let dist = meshlib::find_signed_distance_mr_mesh_part(&MeshPart::construct(&mesh1, None), &MeshPart::construct(&mesh2, None), None, None);
    println!("Signed distance between meshes is {}", dist.signed_dist());

    Ok(())
}
