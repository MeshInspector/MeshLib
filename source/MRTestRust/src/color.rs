use meshlib::mr;

#[test]
fn new()
{
    let color = mr::Color::new();
    assert_eq!((color.r, color.g, color.b, color.a), (0, 0, 0, 255));
}

#[test]
fn from_components()
{
    let color = mr::Color::construct_4_int(128, 64, 32, 200);
    assert_eq!((color.r, color.g, color.b, color.a), (128, 64, 32, 200));
}

#[test]
fn from_float_components()
{
    let color = mr::Color::construct_4_float(0.5, 0.25, 0.125, 1.0);
    assert_eq!((color.r, color.g, color.b, color.a), (127, 63, 31, 255));
}

#[test]
fn get_uint32()
{
    let color = mr::Color::construct_4_int(128, 64, 32, 200);
    assert_eq!(color.get_u_int32(), 0xC8204080);
}

#[test]
fn vert_colors_new_sized()
{
    let colors = mr::VertColors::construct_1_size_t(10);
    assert_eq!(colors.size(), 10);
}
