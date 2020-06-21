[![Actions Status](https://github.com/prinz7/NNApproximator/workflows/Build/badge.svg)](https://github.com/prinz7/NNApproximator/actions)

# NNApproximator
Project to test the approximation of an unknown function with a neural network.

To get the needed library files via git, you need to activate Git LFS in your cloned repository.
```
git lfs install
```

Then pull the library files:
```
git lfs pull
```

#### Build project manually in terminal:
In the cloned folder create a folder named 'build' and enter it:
```
mkdir build
cd build
```

Call CMake to create the make files. You need at least CMake version 3.13 for this to work:
```
cmake ..
```

After that you can build the program with e.g. 4 threads:
```
make -j4
```
