import meshlib.AffineXf3f;
import meshlib.BooleanOperation;
import meshlib.BooleanResult;
import meshlib.Functions;
import meshlib.Matrix3f;
import meshlib.Mesh;
import meshlib.SphereParams;
import meshlib.Vector3f;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class BooleanBenchmark
{
    // Subtracts sphereB rotated around Z-axis by i * 0.1 degrees from sphereA.
    // sphereB itself is not modified, so the AABB trees of both meshes are built once and reused.
    static void subtractRotated(Mesh sphereA, Mesh sphereB, int i)
    {
        AffineXf3f rigidB2A = AffineXf3f.linear(Matrix3f.rotation(Vector3f.plusZ(), (float)Math.toRadians(i * 0.1)));
        BooleanResult result = Functions.boolean_(sphereA, sphereB, BooleanOperation.DifferenceAB, rigidB2A, null, null);
        if (!result.valid())
            throw new RuntimeException("Failed to perform boolean: " + result.errorString());
    }

    public static void main(String[] args)
    {
        // The number of Booleans can be given in the command line.
        int numBooleans = args.length > 0 ? Integer.parseInt(args[0]) : 2000;

        // Two identical spheres of radius 1.
        Mesh sphereA = Functions.makeSphere(new SphereParams(1.0f, 3366));
        Mesh sphereB = new Mesh(sphereA);

        // Untimed warm-up Boolean, it builds the AABB trees.
        subtractRotated(sphereA, sphereB, 1);

        long start = System.nanoTime();
        for (int i = 1; i <= numBooleans; ++i)
            subtractRotated(sphereA, sphereB, i);
        double seconds = (System.nanoTime() - start) * 1e-9;

        System.out.printf("%d Booleans in %.3f s, %.1f Booleans/s%n", numBooleans, seconds, numBooleans / seconds);
    }
}
