using MR.DotNet;
using System;
using System.Reflection;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length != 4)
            Console.WriteLine( "Usage: {0}  {{unite | intersect}}  INPUT1 INPUT2 OUTPUT", Assembly.GetExecutingAssembly().GetName().Name );

        BooleanOperation op;
        switch (args[0])
        {
            case "unite":
                op = BooleanOperation.Union;
                break;
            case "intersect":
                op = BooleanOperation.Intersection;
                break;
            default:
                Console.WriteLine( "Unknown operation: {0}", args[0] );
                return;
        }

        try
        {
            Mesh meshA = Mesh.FromAnySupportedFormat(args[1]);
            Mesh meshB = Mesh.FromAnySupportedFormat(args[2]);
            var res = MeshBoolean.Boolean(meshA, meshB, op);
            Mesh.ToAnySupportedFormat(res.mesh, args[3]);
        }
        catch (Exception e)
        {
            Console.WriteLine( "Error: {0}", e.Message );
        }
    }
}
