use meshlib::{AffineXf3f, BooleanOperation, Matrix3f, Mesh, SphereParams, Vector3f};

// Subtracts sphere_b rotated around Z-axis by i * 0.1 degrees from sphere_a.
// sphere_b itself is not modified, so the AABB trees of both meshes are built once and reused.
fn subtract_rotated(sphere_a: &Mesh, sphere_b: &Mesh, i: i32) -> Result<(), String>
{
    let rigid_b2a = AffineXf3f::linear(&Matrix3f::rotation_float(&Vector3f::plus_z(), (i as f32 * 0.1).to_radians()));
    let result = meshlib::boolean_with_xf(sphere_a, sphere_b, BooleanOperation::DifferenceAB, Some(&rigid_b2a), None, None);
    if !result.valid()
    {
        return Err(format!("Failed to perform boolean: {}", result.error_string()));
    }
    Ok(())
}

fn main() -> Result<(), String>
{
    // The number of Booleans can be given in the command line.
    let num_booleans: i32 = std::env::args().nth(1).map_or(2000, |arg| arg.parse().expect("the number of Booleans"));

    // Two identical spheres of radius 1.
    let sphere_a = meshlib::make_sphere(&SphereParams::construct_from(1.0, 3366));
    let sphere_b = sphere_a.clone();

    // Untimed warm-up Boolean, it builds the AABB trees.
    subtract_rotated(&sphere_a, &sphere_b, 1)?;

    let start = std::time::Instant::now();
    for i in 1..=num_booleans
    {
        subtract_rotated(&sphere_a, &sphere_b, i)?;
    }
    let seconds = start.elapsed().as_secs_f64();

    println!("{num_booleans} Booleans in {seconds:.3} s, {:.1} Booleans/s", num_booleans as f64 / seconds);
    Ok(())
}
