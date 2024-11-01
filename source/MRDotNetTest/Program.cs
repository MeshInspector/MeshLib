using NUnitLite;
using System;

namespace MRDotNetTest2
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting tests...");
            new AutoRun().Execute(args);
        }
    }
}
