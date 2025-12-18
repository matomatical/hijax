import jax
import jax.numpy as jnp
from jaxtyping import Int, Array

print("== LESSON 2: DISTINGUISHING TRACING VS. EXECUTION ======")

print("1. defining and transforming function...")
@jax.jit
def f(
    a: Int[Array, "n"],
    b: Int[Array, "n"],
) -> Int[Array, "n"]:
    print("   [f.trace] a:", a)
    jax.debug.print("   [f.debug] a: {}", a)
    print("   [f.trace] b:", b)
    jax.debug.print("   [f.debug] b: {}", b)
    c = a + b
    print("   [f.trace] ->", c)
    jax.debug.print("   [f.debug] -> {}", c)
    return c
print("   f:", f)

print("2. calling function (tracing and executing)...")
result = f(jnp.arange(8), jnp.ones(8, dtype=int))
print("   result:", result)

print("3. calling function with same types (only executing)...")
result = f(jnp.arange(8), jnp.ones(8, dtype=int))
print("   result:", result)

print("4. calling function with different types (tracing and executing)...")
result = f(jnp.arange(4), jnp.ones(4, dtype=int))
print("   result:", result)
