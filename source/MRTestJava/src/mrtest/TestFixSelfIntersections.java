package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.expected_MR_FaceBitSet_std_string;
import meshlib.mr.selfintersections.Settings;
import meshlib.mr.selfintersections.Settings_Method;

public final class TestFixSelfIntersections
{
    public static void fixSelfIntersections()
    {
        Mesh mesh = Functions.makeTorusWithSelfIntersections(1.0f, 0.2f, 32, 16, null);
        checkEq(mesh.topology().getValidFaces().count(), 1024, "faces");

        expected_MR_FaceBitSet_std_string intersections = meshlib.mr.selfintersections.Functions.getFaces(mesh, false, null);
        check(intersections.success(), "getFaces failed");
        checkEq(intersections.value().count(), 128, "intersections");

        Settings settings = new Settings();
        settings.setMethod(Settings_Method.CutAndFill);
        settings.setTouchIsIntersection(false);
        meshlib.mr.selfintersections.Functions.fix(mesh, settings);

        long validFaces = mesh.topology().getValidFaces().count();
        check(validFaces == 1194 || validFaces == 1196, "valid faces after fix: " + validFaces); // on some macOS Arm runners in Debug mode

        intersections = meshlib.mr.selfintersections.Functions.getFaces(mesh, false, null);
        check(intersections.success(), "getFaces failed");
        checkEq(intersections.value().count(), 0, "no intersections");
    }
}
