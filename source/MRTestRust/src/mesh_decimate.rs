use meshlib::mr;

const PI_F: f32 = 3.14159265358979;

#[test]
fn mesh_decimate()
{
    let mut mesh = mr::make_cylinder_advanced(Some(0.5), Some(0.5), Some(0.0), Some(20.0 / 180.0 * PI_F), Some(1.0), Some(16));

    // Select all faces.
    let mut region = mr::FaceBitSet::construct_from_another(mesh.topology().get_valid_faces());
    let region_saved = region.clone();

    // Setup and run the decimator.
    let result = {
        let mut settings = mr::DecimateSettings::new();
        settings.set_max_error(0.001);
        settings.set_region(Some(&mut region));
        settings.set_max_triangle_aspect_ratio(80.0);
        mr::decimate_mesh(&mut mesh, Some(&settings))
    };

    // Compare the regions and the deleted vertices and faces.
    assert!(!mr::op_equal_mr_face_bit_set(&region_saved, &region));
    assert!(*result.verts_deleted() > 0);
    assert!(*result.faces_deleted() > 0);
}

// Not in MRTestC2: a progress callback made from a Rust closure.
#[test]
fn mesh_decimate_with_progress_callback()
{
    use std::sync::atomic::{AtomicI32, Ordering};
    let mut mesh = mr::make_cylinder_advanced(Some(0.5), Some(0.5), Some(0.0), Some(20.0 / 180.0 * PI_F), Some(1.0), Some(16));
    let calls = std::sync::Arc::new(AtomicI32::new(0));
    let calls2 = calls.clone();
    let mut settings = mr::DecimateSettings::new();
    settings.set_max_error(0.001);
    settings.set_progress_callback(&mr::std_function_bool_from_float::from_fn(move |progress|
    {
        assert!((0.0..=1.0).contains(&progress));
        calls2.fetch_add(1, Ordering::Relaxed);
        true
    }));
    let result = mr::decimate_mesh(&mut mesh, Some(&settings));
    assert!(*result.faces_deleted() > 0);
    assert!(calls.load(Ordering::Relaxed) > 0);

    // Cancelling through the callback.
    let mut mesh = mr::make_cylinder_advanced(Some(0.5), Some(0.5), Some(0.0), Some(20.0 / 180.0 * PI_F), Some(1.0), Some(16));
    settings.progress_callback_mut().assign_fn(|_| false);
    let result = mr::decimate_mesh(&mut mesh, Some(&settings));
    assert!(*result.cancelled());
}
