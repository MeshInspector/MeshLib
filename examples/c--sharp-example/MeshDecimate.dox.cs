using MR.DotNet;
using System;
using System.Reflection;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length != 1 && args.Length != 2)
        {           
            Console.WriteLine("Usage: {0} INPUT [OUTPUT]", Assembly.GetExecutingAssembly().GetName().Name);
            return;
        }

        try
        {
            string input = args[0];
            string output = args.Length == 2 ? args[1] : args[0];

            var mesh = Mesh.FromAnySupportedFormat( input );

            DecimateParameters dp = new DecimateParameters();
            dp.strategy = DecimateStrategy.MinimizeError;
            dp.maxError = 1e-5f * mesh.BoundingBox.Diagonal();
            dp.tinyEdgeLength = 1e-3f;
            dp.packMesh = true;

            var result = MeshDecimate.Decimate(mesh, dp);
            Mesh.ToAnySupportedFormat(mesh, output);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
