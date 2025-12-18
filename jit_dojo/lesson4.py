import jax
import jax.numpy as jnp
from jaxtyping import Int, Array

print("== LESSON 4: LOOPS ARE UNROLLED AT TRACE TIME ==========")

print("1. defining function...")
@jax.jit
def f(a: Int[Array, "n"]) -> Int[Array, "n"]:
    print("   [f.trace] a:", a)
    n, = a.shape
    out = jnp.zeros_like(a)
    for i in range(n):
        print(f"   [f.trace] loop iteration i={i}")
        out = out.at[i].set(a[i] * i)
        
    print("   [f.trace] ->", out)
    return out
print("   f:", f)

print("2. calling function (tracing and executing)...")
result = f(jnp.array([1, 2, 3, 4]))
print("   result:", result)

print("3. calling function again (only executing)...")
result = f(jnp.array([5, 5, 5, 5]))
print("   result:", result)

