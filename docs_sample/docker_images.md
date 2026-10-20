# Docker Images and Layers

A Docker image is a read-only template built from a Dockerfile, and a container is a running instance of it. Every instruction in the Dockerfile produces a layer, and layers are cached: if an instruction and everything before it are unchanged, the builder reuses the stored layer instead of running it again.

That makes instruction order a performance decision. Steps that rarely change, such as installing dependencies, should come first, and steps that change on every commit, such as copying the application source, should come last. Copying only requirements.txt, installing the packages, and then copying the rest of the code means that editing a source file does not invalidate the slow install layer.

Two habits keep images small. A multi-stage build compiles or installs tools in one stage and copies only the finished artifacts into a slim final image. A .dockerignore file keeps things like .git folders and virtual environments out of the build context so they never enter any layer.
