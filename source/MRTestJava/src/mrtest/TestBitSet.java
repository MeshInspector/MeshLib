package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.mr.BitSet;
import meshlib.mr.Functions;

public final class TestBitSet
{
    public static void bitSet()
    {
        BitSet a = new BitSet(10, false);
        checkEq(a.size(), 10, "size");
        a.set(5, true);
        check(a.test(5), "test 5");
        check(!a.test(4), "test 4");
        checkEq(a.find_last(), 5, "find_last");

        BitSet b = new BitSet(0, false);
        b.autoResizeSet(6, true);
        checkEq(b.size(), 7, "autoResizeSet");

        a.resize(10, false);
        a.set(5, true);
        a.set(6, true);
        b.resize(10, false);
        b.set(6, true);
        BitSet c = Functions.opSub(a, b);
        check(c.test(5), "sub 5");
        check(!c.test(6), "sub 6");
    }
}
