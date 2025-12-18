import jax
import jax.numpy as jnp
from jaxtyping import Int, Array

print("== LESSON 7: CAN'T BRANCH ON DYNAMIC VALUES ============")

print("1. defining function...")
def f(a: Int[Array, "n"]) -> Int[Array, "n"]:
    print("   [f.trace] a:", a)
    n, = a.shape
    pos0 = a[0] > 0
    print(f"   [f.trace] pos0={pos0}")
    jax.debug.print("   [f.debug] pos0={}", pos0)
    if n > 0 and pos0:
        print("   [f.trace] ->", a)
        return a
    else:
        print("   [f.trace] ->", -a)
        return -a
print("   f:", f)

print("2. calling function without jit...")
result = f(jnp.array([1, 2, 3, 4]))
print("   result:", result)
result = f(jnp.array([-1, -2, -3, 4]))
print("   result:", result)

print("3. transforming and calling function again...")
f_jit = jax.jit(f)
result = f_jit(jnp.array([5, 5, 5, 5]))
print("   result:", result)
# expect error
