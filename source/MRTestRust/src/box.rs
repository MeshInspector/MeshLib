use meshlib::mr::{Box3f, Box3i, Vector3f, Vector3i};

#[test]
fn boxf()
{
    let b = Box3f::from_min_and_size(&Vector3f::diagonal(-1.0), &Vector3f::diagonal(1.0));
    let volume = b.volume();
    assert!(volume > 0.999 && volume < 1.001);
    let diag = b.diagonal();
    assert!(diag > 1.73 && diag < 1.74);
    assert!(b.valid());
    let size = b.size();
    assert!(size.x > 0.999 && size.x < 1.001);
    assert!(size.y > 0.999 && size.y < 1.001);
    assert!(size.z > 0.999 && size.z < 1.001);
    let center = b.center();
    assert!(center.x < -0.499 && center.x > -0.501);
    assert!(center.y < -0.499 && center.y > -0.501);
    assert!(center.z < -0.499 && center.z > -0.501);
}

#[test]
fn boxf_invalid()
{
    assert!(!Box3f::new().valid());
}

#[test]
fn boxi()
{
    let b = Box3i::from_min_and_size(&Vector3i::diagonal(-2), &Vector3i::diagonal(2));
    assert_eq!(b.volume(), 8);
    assert!(b.valid());
    let size = b.size();
    assert_eq!((size.x, size.y, size.z), (2, 2, 2));
    let center = b.center();
    assert_eq!((center.x, center.y, center.z), (-1, -1, -1));
}

#[test]
fn boxi_invalid()
{
    assert!(!Box3i::new().valid());
}
