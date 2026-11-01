# Boolean Benchmark {#ExampleBooleanBenchmark}

Measures the speed of [boolean](\ref MR::boolean) operations: subtracts from a sphere the same sphere
rotated around Z-axis by an angle increasing by 0.1 degrees each time, and prints the elapsed time
and the number of Booleans per second. The rotation is passed as `rigidB2A`, so neither mesh
is modified and their AABB trees are built only once.

<div class="tabbed">
 
- <b class="tab-title">C++</b>
\include BooleanBenchmark.dox.cpp
 
- <b class="tab-title">Python</b>
\include BooleanBenchmark.dox.py
 
- <b class="tab-title">C</b>
\include BooleanBenchmark.dox.c
 
- <b class="tab-title">C#</b>
\include BooleanBenchmark.dox.cs
 
- <b class="tab-title">JavaScript</b>
\include BooleanBenchmark.dox.js
\include{doc} JsUsingNote.md

- <b class="tab-title">Rust</b>
\include BooleanBenchmark.dox.rs

- <b class="tab-title">Java</b>
\include BooleanBenchmark.dox.java

</div>
