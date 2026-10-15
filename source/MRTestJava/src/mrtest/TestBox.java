package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.mr.Box3f;
import meshlib.mr.Box3i;
import meshlib.mr.Vector3f;
import meshlib.mr.Vector3i;

public final class TestBox
{
    public static void boxf()
    {
        Box3f b = Box3f.fromMinAndSize(Vector3f.diagonal(-1.0f), Vector3f.diagonal(1.0f));
        float volume = b.volume();
        check(volume > 0.999f && volume < 1.001f, "volume");
        float diag = b.diagonal();
        check(diag > 1.73f && diag < 1.74f, "diagonal");
        check(b.valid(), "valid");
        Vector3f size = b.size();
        check(size.x() > 0.999f && size.x() < 1.001f, "size x");
        check(size.y() > 0.999f && size.y() < 1.001f, "size y");
        check(size.z() > 0.999f && size.z() < 1.001f, "size z");
        Vector3f center = b.center();
        check(center.x() < -0.499f && center.x() > -0.501f, "center x");
        check(center.y() < -0.499f && center.y() > -0.501f, "center y");
        check(center.z() < -0.499f && center.z() > -0.501f, "center z");
    }

    public static void boxfInvalid()
    {
        check(!new Box3f().valid(), "invalid");
    }

    public static void boxi()
    {
        Box3i b = Box3i.fromMinAndSize(Vector3i.diagonal(-2), Vector3i.diagonal(2));
        checkEq(b.volume(), 8, "volume");
        check(b.valid(), "valid");
        Vector3i size = b.size();
        check(size.x() == 2 && size.y() == 2 && size.z() == 2, "size");
        Vector3i center = b.center();
        check(center.x() == -1 && center.y() == -1 && center.z() == -1, "center");
    }

    public static void boxiInvalid()
    {
        check(!new Box3i().valid(), "invalid");
    }
}
