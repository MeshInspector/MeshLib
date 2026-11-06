use meshlib::NoiseSettings;

fn main() -> Result<(), String>
{
    // Load mesh
    let mut mesh = meshlib::mesh_load::from_any_supported_format("mesh.stl", None).into_result()?;

    // Add noise to the mesh
    let mut noise_settings = NoiseSettings::new();
    noise_settings.set_sigma(mesh.compute_bounding_box(None).diagonal() * 0.0001);
    let _ = meshlib::add_noise_mr_mesh(&mut mesh, None, Some(&noise_settings));

    // Invalidate the mesh because of the external vertex changes
    mesh.invalidate_caches(None);

    // Save the noised mesh
    meshlib::mesh_save::to_any_supported_format(&mesh, "noised_mesh.stl", None).into_result()?;

    // Denoise the mesh with sharpening for sharp edges
    // see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
    meshlib::mesh_denoise_via_normals(&mut mesh, None);

    // Save the denoised mesh
    meshlib::mesh_save::to_any_supported_format(&mesh, "denoised_mesh.stl", None).into_result()?;

    Ok(())
}
