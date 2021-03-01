# A software implementation of Optix 7

Based on: Siggraph 2019/2020 OptiX 7/7.1 Course Tutorial Code


# About this repository  

This repository implements Optix in Software. It was written to learn the Optix API.
It mainly consists of parts of an pathtracer I wrote (https://github.com/gonsolo/gonzales)
as rendering engine and glue code implementing the Optix API.
The API was implemented step by step, starting with example 01 and progressively extended
as can be seen in the history.

This repository was forked from the original course at https://gitlab.com/ingowald/optix7course.git.

# Building the Code

"make" builds everything. Then go to an example and type "make run".
I only tested it on Linux.

## Dependencies

- a C++ compiler (gcc 10.2)
- a Swift compiler (swiftc 5.3.3).
- no Optix SDK, Cuda or Nvidia drivers are needed. ;)
- GLFW

## Building under Linux

- Install required packages

    - on Ubuntu: `sudo apt install libglfw3-dev`

- Clone the code
```
    git clone https://github.com/gonsolo/optix7course
    cd optix7course
```

- build
```
    make
```

- run an example
```
    cd example01_helloOptix; make run
```

# Examples Overview
	
## Example 1: Hello World 

This is simplest example and only needs a few lines to implement.

This is how it looks on my Linux machine:
![Example 1](./example01_helloOptix/example.01.png)


## Example 2: First Pipeline Setup and Raygen Program

![Example 2](./example02_pipelineAndRayGen/example.02.png)

<!--

## Example 3: Rendering in a GLFW Window 

Rendering to files is nice and well, but *probably* you want to
eventually do some online rendering; so this example moves the
previous raygen example into a 3D viewer window (created and run using
GLFW). For now this viewer just displays the rendered images, with no
user interaction.

![Same Raygen example, in GLFL Window (Linux)](./example03_inGLFWindow/ex03-linux.png)
![Same Raygen example, in GLFL Window (Windows)](./example03_inGLFWindow/ex03-windows.png)

## Example 4: Creating a first Triangle Mesh and Accel Struct 

Though the previous setup steps were important to get right, eventually 
you want to use a ray tracer to trace some real rays against some
real geometry. 

This example introduces how to create some Triangle Mesh Geometry (in
this example, two simple, hardcoded, cubes), how to build an
Acceleration Structure over this "BuildInput", and how to trace rays
against it. To do this we also need to introduce a simple camera model.

![First Triangle Mesh and Accel Struct](./example04_firstTriangleMesh/ex04.png)

## Example 5: First Shader Binding Table (SBT) Data 

The earlier examples *created* an SBT (they had to, else they couldn't
have executed any OptiX launch), but didn't actually put any data into
the SBT. This example introduces how to do that, by putting just some
simple constant per-object color into the mesh's SBT entry, then shading
it based on the surface normal's angle to the view ray.

![First SBT Data](./example05_firstSBTData/ex05.png)

## Example 6: Multiple Triangle Meshes 

This example introduces the concept of having multiple different
meshes (each with their own programs and SBT data) into the same accel
structure. Whereas the previous example used two (same color) cubes in
*one* triangle mesh, this example split this test scene into two
meshes with one cube (and one color) each.

![Multiple Triangle Meshes](./example06_multipleObjects/ex06.png)

## Example 7: First Real Model

This example takes the previous "multiple meshes" code unmodified, but
introduces a simple OBJ file format parser (using [Syoyo Fuyita's
tinyobj](https://github.com/syoyo/tinyobjloader), and hooks the resulting triangle meshes up to
the previous example's render code.

For this example, you must download the [Crytek Sponza model](https://casual-effects.com/data/) and unzip it to the (non-existent, until you create it) subdirectory `optix7course/models`.

And la-voila, with exactly the same render code from Sample 6, it
suddenly starts to take shape:

![First Real Model: Sponza](./example07_firstRealModel/ex07.png)

## Example 8: Adding Textures via CUDA Texture Objects

This example shows how to create and set up CUDA texture objects on
the host, with the host passing those to the device via the SBT, and how to use
those texture objects on the device. This one will take a bit of time
to load in Debug - it's worth the wait! Or simply build and run in Release.

![Adding Textures](./example08_addingTextures/ex08.png)

## Example 9: Adding a second ray type: Shadows

This is the last example that focuses on host-side setup, in this
case adding a second ray type (for shadow rays), which also requires
changing the way the SBT is being built. 

This sample also shows how to shoot secondary rays (the shadow rays)
in device programs, how to use an any-hit program for the shadow rays,
how to call *optixTerminateRay* from within an any-hit program, and how
to use the optixTrace call's SBT index/offset values to specify the
ray type.

![Adding Shadow Rays](./example09_shadowRays/ex09.png)

## Example 10: Soft Shadows

Whereas the first 9 examples focused on how to perform all the
required host-side setup for various incremental features, this
example can now start to focus more on the "ray tracing 101" style
additions that focus what rays to trace to add certain rendering
effects. 

This simple example intentionally only adds soft shadows from area
lights, but extending this to add reflections, refraction, diffuse
bounces, better material models/BRDFs, etc., should from now on be
straightforward. 

Please feel free to play with adding these examples ... and share what
you did!

![Soft Shadows](./example10_softShadows/ex10.png)

-->
