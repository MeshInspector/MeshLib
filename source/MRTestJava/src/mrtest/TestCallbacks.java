package mrtest;

import static mrtest.Common.check;

import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.MeshPart;
import meshlib.mr.Object;
import meshlib.mr.ObjectMesh;
import meshlib.mr.OffsetParameters;
import meshlib.mr.std_function_bool_from_float;

/** The `std::function` callbacks from Java lambdas, and the casts between a class and its bases. */
public final class TestCallbacks
{
    public static void progressCallback()
    {
        Mesh mesh = Common.createCube();
        MeshPart mp = new MeshPart(mesh, null);
        OffsetParameters params = new OffsetParameters();
        params.setVoxelSize(Functions.suggestVoxelSize(mp, 10000.0f));
        int[] calls = {0};
        float[] last = {-1.0f};
        params.setCallBack(std_function_bool_from_float.of(progress ->
        {
            calls[0]++;
            check(progress >= last[0] && progress <= 1.0f, "progress goes up: " + progress);
            last[0] = progress;
            return true;
        }));
        check(Functions.offsetMesh(mp, 0.1f, params).success(), "offsetMesh");
        check(calls[0] > 0, "the callback was called");
    }

    public static void cancellingCallback()
    {
        Mesh mesh = Common.createCube();
        MeshPart mp = new MeshPart(mesh, null);
        OffsetParameters params = new OffsetParameters();
        params.setVoxelSize(Functions.suggestVoxelSize(mp, 10000.0f));
        params.setCallBack(std_function_bool_from_float.of(progress -> false));
        check(!Functions.offsetMesh(mp, 0.1f, params).success(), "cancelled");
    }

    public static void callbackException()
    {
        std_function_bool_from_float f = std_function_bool_from_float.of(progress -> {throw new IllegalStateException("from the lambda");});
        boolean caught = false;
        try
        {
            f.call(0.5f);
        }
        catch (IllegalStateException e)
        {
            caught = e.getMessage().equals("from the lambda");
        }
        check(caught, "the exception of the lambda is rethrown");
    }

    public static void casts()
    {
        ObjectMesh objectMesh = new ObjectMesh();
        Object base = objectMesh.upcastToObject();
        check(!base.isOwned(), "the upcast is a reference");
        ObjectMesh back = ObjectMesh.dynamicDowncastFromObject(base);
        check(back != null && back.rawPointer().address() == objectMesh.rawPointer().address(), "dynamic downcast");
        check(ObjectMesh.staticDowncastFromObject(base).rawPointer().address() == objectMesh.rawPointer().address(), "static downcast");
        meshlib.mr.ObjectPoints notAMesh = new meshlib.mr.ObjectPoints();
        check(ObjectMesh.dynamicDowncastFromObject(notAMesh.upcastToObject()) == null, "dynamic downcast of another class is null");
    }
}
