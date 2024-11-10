using MR.DotNet;
using System;
using System.Globalization;
using System.Reflection;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length != 2 && args.Length != 3)
        {           
            Console.WriteLine("Usage: {0} OFFSET_VALUE INPUT [OUTPUT]", Assembly.GetExecutingAssembly().GetName().Name);
            return;
        }

        try
        {
            float offsetValue = float.Parse(args[0],
                      System.Globalization.NumberStyles.AllowThousands,
                      CultureInfo.InvariantCulture);

            string input = args[1];
            string output = args.Length == 3 ? args[2] : args[1];

            MeshPart mp = new MeshPart();
            mp.mesh = Mesh.FromAnySupportedFormat( args[1] );

            OffsetParameters op = new OffsetParameters();
            op.voxelSize = Offset.SuggestVoxelSize(mp, 1e6f);

            var result = Offset.OffsetMesh(mp, offsetValue, op);

            Mesh.ToAnySupportedFormat(result, output);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
