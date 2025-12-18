import jax
import jax.numpy as jnp
from jaxtyping import Int, Array

print("== LESSON 1: JUST-IN-TIME COMPILATION ==================")

print("1. defining function...")
def f_nojit(
    a: Int[Array, "n"],
    b: Int[Array, "n"],
) -> Int[Array, "n"]:
    print("   [f] a:", a)
    print("   [f] b:", b)
    c = a + b
    print("   [f] ->", c)
    return c
print("   f_nojit:", f_nojit)

print("2. transforming function...")
f_jit = jax.jit(f_nojit)
print("   f_jit:", f_jit)

print("3. calling function...")
result = f_jit(jnp.arange(8), jnp.ones(8, dtype=int))
print("   result:", result)
