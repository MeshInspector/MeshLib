import { ml, check, checkEq } from "./common.mjs";

export function bitSet()
{
    const a = new ml.BitSet(10, false);
    checkEq(a.size(), 10, "size");
    a.set(5, true);
    check(a.test(5), "test 5");
    check(!a.test(4), "test 4");
    checkEq(a.find_last(), 5, "find_last");
    const b = new ml.BitSet(0, false);
    b.autoResizeSet(6, true);
    checkEq(b.size(), 7, "autoResizeSet");
    a.resize(10, false);
    a.set(5, true);
    a.set(6, true);
    b.resize(10, false);
    b.set(6, true);
    const c = ml.opSub(a, b);
    check(c.test(5), "sub 5");
    check(!c.test(6), "sub 6");
}

export function typedBitSet()
{
    const faces = new ml.FaceBitSet();
    faces.resize(12, true);
    for (let i = 6; i < 12; i++)
        faces.set(new ml.FaceId(i), false);
    checkEq(faces.count(), 6, "count");
    check(faces.test(new ml.FaceId(0)) && !faces.test(new ml.FaceId(6)), "test");
    const copy = new ml.FaceBitSet(faces);
    copy.set(new ml.FaceId(6), true);
    checkEq(faces.count(), 6, "the copy is independent");
    check(!ml.opEqual(faces, copy), "not equal");
    check(ml.opEqual(faces, new ml.FaceBitSet(faces)), "equal");
    // Ownership and explicit deletion.
    check(faces.isOwned(), "owned");
    faces.delete();
    let threw = false;
    try
    {
        faces.count();
    }
    catch (e)
    {
        threw = true;
    }
    check(threw, "deleted");
}
