import functools
import textwrap
import jax
import jax.numpy as jnp
from jaxtyping import Int, Array

print("== LESSON 6: ONLY SOME TYPES CAN BE STATIC/DYNAMIC =====")

print("1. defining and transforming function...")
@functools.partial(jax.jit, static_argnames=["n"])
def f(
    n: int,
    a: Int[Array, "m"],
) -> Int[Array, "n"]:
    print("   [f.trace] n:", n)
    print("   [f.trace] a:", a)
    jax.debug.print("   [f.debug] n: {}", n)
    jax.debug.print("   [f.debug] a: {}", a)
    return n + a
print("   f:", f)

print("2. calling function (tracing and executing)...")
result = f(
    n=10,
    a=jnp.arange(5),
)
print("   result:", result)

print("3. calling function again (tracing and executing)...")
result = f(
    n=11,
    a=jnp.arange(5),
)
print("   result:", result)

print("4. only hashable types can be passed as static params")
try:
    f(
        n=jnp.array(1),
        a=jnp.arange(5),
    )
except Exception as e:
    print("  > "+"\n  > ".join(textwrap.wrap(str(e))))

print("5. only array(tree) types can be passed as dynamic params")
try:
    f(
        n=(min, max),
        a=(sum,),
    )
except Exception as e:
    print("  > "+"\n  > ".join(textwrap.wrap(str(e))))
