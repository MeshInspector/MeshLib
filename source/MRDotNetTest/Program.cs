using NUnitLite;
using System;

namespace MR.Test
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
