package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.mr.Color;
import meshlib.mr.VertColors;

public final class TestColor
{
    static boolean rgba(Color color, int r, int g, int b, int a)
    {
        return (color.r() & 0xFF) == r && (color.g() & 0xFF) == g && (color.b() & 0xFF) == b && (color.a() & 0xFF) == a;
    }

    public static void construct()
    {
        check(rgba(new Color(), 0, 0, 0, 255), "default");
    }

    public static void fromComponents()
    {
        check(rgba(new Color(128, 64, 32, 200), 128, 64, 32, 200), "int components");
    }

    public static void fromFloatComponents()
    {
        check(rgba(new Color(0.5f, 0.25f, 0.125f, 1.0f), 127, 63, 31, 255), "float components");
    }

    public static void getUint32()
    {
        checkEq(new Color(128, 64, 32, 200).getUInt32(), 0xC8204080, "getUInt32");
    }

    public static void vertColorsSized()
    {
        checkEq(new VertColors(10).size(), 10, "size");
    }
}
