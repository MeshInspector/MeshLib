package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.FaceBitSet;
import meshlib.Functions;
import meshlib.Mesh;
import meshlib.selfintersections.Settings;
import meshlib.selfintersections.Settings_Method;

public final class TestFixSelfIntersections
{
    public static void fixSelfIntersections()
    {
        Mesh mesh = Functions.makeTorusWithSelfIntersections(1.0f, 0.2f, 32, 16, null);
        checkEq(mesh.topology().getValidFaces().count(), 1024, "faces");

        FaceBitSet intersections = meshlib.selfintersections.Functions.getFaces(mesh, false, null);
        checkEq(intersections.count(), 128, "intersections");

        Settings settings = new Settings();
        settings.setMethod(Settings_Method.CutAndFill);
        settings.setTouchIsIntersection(false);
        meshlib.selfintersections.Functions.fix(mesh, settings);

        long validFaces = mesh.topology().getValidFaces().count();
        check(validFaces == 1194 || validFaces == 1196, "valid faces after fix: " + validFaces); // on some macOS Arm runners in Debug mode

        intersections = meshlib.selfintersections.Functions.getFaces(mesh, false, null);
        checkEq(intersections.count(), 0, "no intersections");
    }
}
