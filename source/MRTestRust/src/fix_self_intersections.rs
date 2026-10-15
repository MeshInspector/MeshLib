use meshlib::mr;
use meshlib::mr::self_intersections;

#[test]
fn fix_self_intersections()
{
    let mut mesh = mr::make_torus_with_self_intersections(Some(1.0), Some(0.2), Some(32), Some(16), None);
    assert_eq!(mesh.topology().get_valid_faces().count(), 1024);

    let mut intersections = self_intersections::get_faces(&mesh, Some(false), None);
    assert_eq!(intersections.value_mut().expect("getFaces failed").count(), 128);

    let mut settings = self_intersections::Settings::new();
    settings.set_method(self_intersections::Settings_Method::CutAndFill);
    settings.set_touch_is_intersection(false);
    self_intersections::fix(&mut mesh, &settings);

    let valid_faces = mesh.topology().get_valid_faces().count();
    assert!(valid_faces == 1194 || valid_faces == 1196); // on some macOS Arm runners in Debug mode

    let mut intersections = self_intersections::get_faces(&mesh, Some(false), None);
    assert_eq!(intersections.value_mut().expect("getFaces failed").count(), 0);
}
