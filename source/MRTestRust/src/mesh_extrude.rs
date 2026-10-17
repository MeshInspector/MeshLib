use crate::common::make_torus;
use meshlib::FaceId;

// The region contains selected faces.
#[test]
fn degenerate_band_non_empty()
{
    let mut mesh = make_torus(1.1, 0.5, 16, 16);
    let count_before = mesh.points().size();
    let mut region = meshlib::FaceBitSet::new();
    region.auto_resize_set(FaceId {id_: 0}, Some(true));
    region.auto_resize_set(FaceId {id_: 1}, Some(true));
    meshlib::make_degenerate_band_around_region(&mut mesh, &region, None);
    mesh.invalidate_caches(Some(true));
    // The number of vertices grows because of the vertex duplication along the boundary.
    assert_ne!(mesh.points().size(), count_before);
}

// The region is empty.
#[test]
fn degenerate_band_empty()
{
    let mut mesh = make_torus(1.1, 0.5, 16, 16);
    let count_before = mesh.points().size();
    let region = meshlib::FaceBitSet::new();
    meshlib::make_degenerate_band_around_region(&mut mesh, &region, None);
    mesh.invalidate_caches(Some(true));
    assert_eq!(mesh.points().size(), count_before);
}
