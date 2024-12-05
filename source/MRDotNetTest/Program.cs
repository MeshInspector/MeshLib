using NUnitLite;
using System;

namespace MR.Test
{
    internal class Program
    {
        static int Main(string[] args)
        {
            Console.WriteLine("Starting tests...");
            return new AutoRun().Execute(args);
        }
    }
}
