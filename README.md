<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Parallel Constraint Application with MPI</h1>

<p>This project demonstrates a parallel implementation of constraint application using MPI for inter-process communication and OpenMP for multi-threading within each process. 
  The <code>primaryEquation</code> object is generated on the root process and broadcast to all other processes to ensure consistency across the distributed system. 
  At this stage, application only supports matrices given in Matrix Market <code>(.mtx)</code> file format</p>

<h2>Requirements</h2>
<ul>
    <li><strong>MPI Library</strong>: An implementation of the Message Passing Interface (MPI), such as Open MPI or MPICH.</li>
    <li><strong>OpenMP</strong>: A multi-platform shared memory multiprocessing API.</li>
</ul>
<h2>Files</h2>
<ul>
    <li><code>test_parallel.cpp</code>: The main program file containing the parallel constraint application implementation.</li>
    <li><code>test_serial.cpp</code>: The main program file containing the serial constraint application implementation.</li>
    <li><code>Matrix.h</code>: Header file for matrix operations.</li>
    <li><code>constraint_app_parallel.h</code>: Header file for the primaryEquation, TransformationMatrix classes and other utilities.</li>
</ul>

<h2>Compilation</h2>
<p>To compile the test programs, you can use either <code>g++</code> or <code>mpic++</code>:</p>

<h3>Using <code>g++</code> for serial implementation</h3>
<pre><code>g++ -o test_serial test_serial.cpp</code></pre>

<h3>Using <code>mpic++</code> for parallel implementation</h3>
<pre><code>mpic++ -fopenmp -o test_parallel test_parallel.cpp</code></pre>

<h2>Running the Program</h2>
<p>To run the parallel program, use the <code>mpirun</code> command with the desired number of processes:</p>
<pre><code>mpirun -np &lt;number_of_processes&gt; ./test_parallel &lt;matrix_folder&gt; &lt;number_of_constraints&gt;</code></pre>
<p>Replace <code>&lt;number_of_processes&gt;</code> with the number of MPI processes you want to launch.</p>
<p>To run the serial program, use runnable object with the matrix folder name and desired number of constraints:</p>
<pre><code>./test_serial &lt;matrix_folder&gt; &lt;number_of_constraints&gt;</code></pre>

<h3>Example</h3>
<pre><code>mpirun -np 4 ./test_parallel bcsstk21 100</code></pre>
<p>This example runs the parallel program with 4 MPI processes using bcsstk21.mtx matrix file and applies 100 random constraints.</p>


