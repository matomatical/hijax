import jax
import jax.numpy as jnp
from jaxtyping import Num, Array

print("== LESSON 3: SHAPE/DTYPES ARE STATICALLY AVAILABLE =====")

print("1. defining and transforming function...")
@jax.jit
def f(a: Num[Array, "n"]) -> Num[Array, "n n"]:
    print("   [f.trace] a:", a)
    print("   [f.trace] a.shape:", a.shape)
    print("   [f.trace] a.dtype:", a.dtype)
    n, = a.shape
    A = jnp.zeros((n, n), dtype=a.dtype)
    A = A.at[jnp.arange(n), jnp.arange(n)].set(a)
    print("   [f.trace] ->", A)
    jax.debug.print("   [f.debug] -> [{} ... {}]", A[0], A[-1])
    return A
print("   f:", f)

print("2. calling function (tracing and executing)...")
result = f(jnp.array([1,2,3,4]))
print("   result:", *str(result).splitlines(), sep="\n    ")

print("3. calling function again with a different dtype...")
result = f(jnp.array([1.,2.,3.,4.]))
print("   result:", *str(result).splitlines(), sep="\n    ")
