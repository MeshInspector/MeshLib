package mrtest;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;

/**
 * Runs the tests of the generated Java bindings, ported from MRTestC2. Every public static method of the test classes is a test.
 * No test framework, to keep the bindings free of dependencies.
 */
public final class Main
{
    private static final Class<?>[] TEST_CLASSES = {
        TestAddNoise.class,
        TestAffineXf.class,
        TestBitSet.class,
        TestBox.class,
        TestColor.class,
        TestExpandShrink.class,
        TestFixSelfIntersections.class,
        TestMatrix3.class,
        TestMesh.class,
        TestMeshBoolean.class,
        TestMeshBuilder.class,
        TestMeshCollide.class,
        TestMeshCollidePrecise.class,
        TestMeshComponents.class,
        TestMeshDecimate.class,
        TestMeshExtrude.class,
        TestMeshFillHole.class,
        TestMeshMeshDistance.class,
        TestMeshNormals.class,
        TestMeshOffset.class,
        TestMeshSubdivide.class,
        TestMeshTopology.class,
        TestPointCloud.class,
        TestPointsToMeshProjector.class,
        TestSimpleObjects.class,
        TestVdbConversions.class,
        TestVector3.class,
    };

    public static void main(String[] args)
    {
        List<String> failures = new ArrayList<>();
        int passed = 0;
        for (Class<?> cl : TEST_CLASSES)
        {
            for (Method method : cl.getDeclaredMethods())
            {
                if (!Modifier.isPublic(method.getModifiers()) || !Modifier.isStatic(method.getModifiers()) || method.getParameterCount() != 0)
                    continue;
                String name = cl.getSimpleName() + "." + method.getName();
                try
                {
                    method.invoke(null);
                    passed++;
                    System.out.println("ok   " + name);
                }
                catch (InvocationTargetException e)
                {
                    failures.add(name);
                    System.out.println("FAIL " + name);
                    e.getCause().printStackTrace(System.out);
                }
                catch (IllegalAccessException e)
                {
                    throw new RuntimeException(e);
                }
            }
        }
        System.out.println(passed + " passed, " + failures.size() + " failed" + (failures.isEmpty() ? "" : ": " + String.join(", ", failures)));
        System.exit(failures.isEmpty() ? 0 : 1);
    }
}
