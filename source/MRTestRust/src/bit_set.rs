
#[test]
fn bit_set()
{
    let mut a = meshlib::BitSet::construct(10, Some(false));
    assert_eq!(a.size(), 10);
    a.set(5, true);
    assert!(a.test(5));
    assert!(!a.test(4));
    assert_eq!(a.find_last(), 5);

    let mut b = meshlib::BitSet::construct(0, Some(false));
    b.auto_resize_set(6, Some(true));
    assert_eq!(b.size(), 7);

    a.resize(10, Some(false));
    a.set(5, true);
    a.set(6, true);
    b.resize(10, Some(false));
    b.set(6, true);
    let c = meshlib::op_sub_mr_bit_set(&a, &b);
    assert!(c.test(5));
    assert!(!c.test(6));
}
