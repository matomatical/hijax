import jax
import jax.numpy as jnp
from jaxtyping import Int, Array

print("== LESSON 5: STATIC PARAMETERS =========================")

print("1. defining function...")
def f_nojit(n: int, ) -> Int[Array, "n n"]:
    print("   [f.trace] n:", n)
    jax.debug.print("   [f.debug] n: {}", n)
    a = jnp.arange(n) + 1
    print("   [f.trace] a:", a)
    jax.debug.print("   [f.debug] a: {}", a)
    b = jnp.diag(a)
    print("   [f.trace] ->", b)
    jax.debug.print("   [f.debug] -> [{} ... {}]", b[0], b[-1])
    return b
print("   f_nojit:", f_nojit)

print("2. transforming function...")
f_jit = jax.jit(f_nojit, static_argnames=['n'])
# NOTE: without 'static_argnames' it raises an error
print("   f_jit:", f_jit)

print("3. calling function (tracing and executing)...")
result = f_jit(4)
print("   result:", *str(result).splitlines(), sep="\n  ")

print("4. calling function again (tracing and executing)...")
result = f_jit(3)
print("   result:", *str(result).splitlines(), sep="\n  ")
