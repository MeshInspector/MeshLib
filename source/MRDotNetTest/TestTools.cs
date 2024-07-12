//#define GENERATE_PATTERNS
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace MR.DotNet.Test
{
    internal class TestTools
    {
        public static string GetPathToTestFile(string path)
        {
            return Directory.GetCurrentDirectory() + "\\..\\..\\..\\MeshLib\\source\\MRDotNetTest\\TestFiles\\" + path;            
        }

        public static string GetPathToPattern(string path)
        {
            return Directory.GetCurrentDirectory() + "\\..\\..\\..\\MeshLib\\source\\MRDotNetTest\\Patterns\\" + path;
        }

        public static bool AreMeshesEqual( Mesh actual, string filePath )
        {
#if GENERATE_PATTERNS
            Mesh.ToFile( actual, filePath );
            return true;
#else
            Mesh expected = Mesh.FromFile( filePath );
            return actual == expected;
#endif
        }
    }
}
