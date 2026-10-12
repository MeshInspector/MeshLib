use meshlib::mr;

/// Creates a cube mesh with diagonal = 1.
pub fn create_cube() -> meshlib::Owned<mr::Mesh>
{
    mr::make_cube(Some(&mr::Vector3f::diagonal(1.0)), Some(&mr::Vector3f::diagonal(-0.5)))
}

/// Creates a torus with the given radii and resolutions.
pub fn make_torus(primary_radius: f32, secondary_radius: f32, primary_resolution: i32, secondary_resolution: i32) -> meshlib::Owned<mr::Mesh>
{
    mr::make_torus(Some(primary_radius), Some(secondary_radius), Some(primary_resolution), Some(secondary_resolution), None)
}

#[macro_export]
macro_rules! assert_approx
{
    ($val:expr, $exp:expr, $eps:expr) =>
    {{
        let (val, exp, eps) = ($val as f64, $exp as f64, $eps as f64);
        assert!((val - exp).abs() <= eps, "expected {exp} (delta {eps}), got {val} (delta {})", (val - exp).abs());
    }};
}
