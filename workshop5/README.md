Workshop 5: Just-in-time compilation with JAX (part 1)
-------------------------------------------------------------------------------

Preliminaries:

* Any questions from the last week?
* Installations: No new installations.
* Data: Same data as last week.

Notes:

* jax.jit is powered by XLA ('ACCelerated Linear Algebra')
  * https://github.com/openxla/xla
  * I don't know much about the details
* The same compiler is available for PyTorch through the PyTorch/XLA
  library (remember the TPU workshop).
* In my opinion JIT fits much more nicely into JAX's 'functional function
  transformation library' paradigm than it does into PyTorch.

Today's workshop files and plan:

```
workshop5/
├── jit_dojo.py           # start by stepping through simple examples
├── mattplotlib.py
├── mnist.npz
├── simplenet.py          # then take last week's SimpLeNet code
└── simplenet_jit.py      # and accelerate it using jax.jit.
```

Challenge:

* Implement a bigger ConvNet using equinox/jit.
  * Consider highlights from the history of deep computer vision:
    * AlexNet (Krizhevsky, Sutskever, and Hinton, 2012)
    * VGGNet (Simonyan and Zisserman, 2015)
    * Inception (Szegedy et al., 2015)
    * ResNet (He et al., 2015)
  * How far can jit take your modern laptop/GPU/TPU?
