using MR.DotNet;
// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");
Vector3f a = new Vector3f(1, 2, 3);
Vector3f b = new Vector3f(4, 5, 6);
var c = a + b;
Console.WriteLine( "{0} {1} {2}", c.x, c.y, c.z );

