using NUnitLite;
using System;

namespace MR.DotNet.Test
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
