use meshlib::{MeshPart, Vector3f};

fn main() -> Result<(), String>
{
    let mesh1 = meshlib::mesh_load::from_any_supported_format("mesh1.ctm", None).into_result()?;
    let point = Vector3f {x: 1.5, y: 1.0, z: 0.5};

    let dist = meshlib::find_signed_distance_mr_vector3f(&point, &MeshPart::construct(&mesh1, None), None, None);
    if let Some(dist) = dist.value()
    {
        println!("Signed distance from point to mesh {}", dist.dist());
    }

    Ok(())
}
