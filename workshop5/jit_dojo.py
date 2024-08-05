"""
jit dojo: lessons on just-in-time compilation in JAX.

Lessons:

0. Start with a plain python function using jax.numpy operations.
1. Use `jax.jit` to compile the function.
2. Distinguishing the different phases of jit: trace, compile, execute.
3. Static parameters are available at trace time.
4. Shapes/dtypes of dynamic parameters are available at trace time.
5. Branching and looping happen at trace time.
6. Restrictions on values passed to static and dynamic parameters.
7. Automatic parameter classification with `equinox.filter_jit`.
"""

import functools
import textwrap
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Num, Int, Array
import equinox as eqx


def main(lesson: Literal[0,1,2,3,4,5,6,7]):
    match lesson:
        case 0: lesson0()
        case 1: lesson1()
        case 2: lesson2()
        case 3: lesson3()
        case 4: lesson4()
        case 5: lesson5()
        case 6: lesson6()
        case 7: lesson7()


def lesson0():
    print("== LESSON 0: PLAIN JAX FUNCTION ========================")

    # print("1. defining function...")
    # def f0_nojit(
    #     a: Int[Array, "n"],
    #     b: Int[Array, "n"],
    # ) -> Int[Array, "n"]:
    #     print("   [f0] a:", a)
    #     print("   [f0] b:", b)
    #     c = a + b
    #     print("   [f0] ->", c)
    #     return c
    # print("   f0_nojit:", f0_nojit)

    # print("2. calling function...")
    # result = f0_nojit(jnp.arange(8), jnp.ones(8, dtype=int))
    # print("   result:", result)


def lesson1():
    print("== LESSON 1: JUST-IN-TIME COMPILATION ==================")

    # print("1. defining function...")
    # def f1_nojit(
    #     a: Int[Array, "n"],
    #     b: Int[Array, "n"],
    # ) -> Int[Array, "n"]:
    #     print("   [f1] a:", a)
    #     print("   [f1] b:", b)
    #     c = a + b
    #     print("   [f1] ->", c)
    #     return c
    # print("   f1_nojit:", f1_nojit)

    # print("2. transforming function...")
    # f1_jit = jax.jit(f1_nojit)
    # print("   f1_jit:", f1_jit)

    # print("3. calling function...")
    # result = f1_jit(jnp.arange(8), jnp.ones(8, dtype=int))
    # print("   result:", result)


def lesson2():
    print("== LESSON 2: DISTINGUISHING TRACING VS. EXECUTION ======")

    # print("1. defining and transforming function...")
    # @jax.jit
    # def f2(
    #     a: Int[Array, "n"],
    #     b: Int[Array, "n"],
    # ) -> Int[Array, "n"]:
    #     print("   [f2.trace] a:", a)
    #     jax.debug.print("   [f2.debug] a: {}", a)
    #     print("   [f2.trace] b:", b)
    #     jax.debug.print("   [f2.debug] b: {}", b)
    #     c = a + b
    #     print("   [f2.trace] ->", c)
    #     jax.debug.print("   [f2.debug] -> {}", c)
    #     return c
    # print("   f2:", f2)

    # print("2. calling function (tracing and executing)...")
    # result = f2(jnp.arange(8), jnp.ones(8, dtype=int))
    # print("   result:", result)

    # print("3. calling function with same types (only executing)...")
    # result = f2(jnp.arange(8), jnp.ones(8, dtype=int))
    # print("   result:", result)

    # print("4. calling function with different types (tracing and executing)...")
    # result = f2(jnp.arange(4), jnp.ones(4, dtype=int))
    # print("   result:", result)


def lesson3():
    print("== LESSON 3: STATIC PARAMETERS =========================")

    # print("1. defining function...")
    # def f3_nojit(n: int, ) -> Int[Array, "n n"]:
    #     print("   [f3.trace] n:", n)
    #     jax.debug.print("   [f3.debug] n: {}", n)
    #     a = jnp.arange(n) + 1
    #     print("   [f3.trace] a:", a)
    #     jax.debug.print("   [f3.debug] a: {}", a)
    #     b = jnp.diag(a)
    #     print("   [f3.trace] ->", b)
    #     jax.debug.print("   [f3.debug] -> [{} ... {}]", b[0], b[-1])
    #     return b
    # print("   f3_nojit:", f3_nojit)

    # print("2. transforming function...")
    # f3_jit = jax.jit(f3_nojit, static_argnames=['n'])
    # # NOTE: without 'static_argnames' it raises an error
    # print("   f3_jit:", f3_jit)

    # print("3. calling function (tracing and executing)...")
    # result = f3_jit(4)
    # print("   result:", *str(result).splitlines(), sep="\n  ")

    # print("4. calling function again (tracing and executing)...")
    # result = f3_jit(3)
    # print("   result:", *str(result).splitlines(), sep="\n  ")


def lesson4():
    print("== LESSON 4: SHAPE/DTYPES ARE STATICALLY AVAILABLE =====")

    # print("1. defining and transforming function...")
    # @jax.jit
    # def f4(a: Num[Array, "n"]) -> Num[Array, "n n"]:
    #     print("   [f4.trace] a:", a)
    #     print("   [f4.trace] a.shape:", a.shape)
    #     print("   [f4.trace] a.dtype:", a.dtype)
    #     n, = a.shape
    #     A = jnp.zeros((n, n), dtype=a.dtype)
    #     A = A.at[jnp.arange(n), jnp.arange(n)].set(a)
    #     print("   [f4.trace] ->", A)
    #     jax.debug.print("   [f4.debug] -> [{} ... {}]", A[0], A[-1])
    #     return A
    # print("   f4:", f4)

    # print("2. calling function (tracing and executing)...")
    # result = f4(jnp.array([1,2,3,4]))
    # print("   result:", *str(result).splitlines(), sep="\n    ")

    # print("3. calling function again with a different dtype...")
    # result = f4(jnp.array([1.,2.,3.,4.]))
    # print("   result:", *str(result).splitlines(), sep="\n    ")


def lesson5():
    print("== LESSON 5: BRANCHING/LOOPING HAPPENS AT TRACE TIME ===")

    # print("1. defining function...")
    # @jax.jit
    # def f5(a: Int[Array, "n"]) -> Int[Array, "n n"]:
    #     print("   [f5.trace] a:", a)
    #     print("   [f5.trace] a.shape:", a.shape)
    #     print("   [f5.trace] a.dtype:", a.dtype)
    #     n, = a.shape
    #     A = jnp.zeros((n, n), dtype=a.dtype)
    #     for i in range(n):
    #         print("   [f5.trace] i:", i)
    #         if i % 2 == 0:
    #             A = A.at[i,i].set(a[i])
    #         else:
    #             A = A.at[i,i].set(-a[i])
    #     print("   [f5.trace] ->", A)
    #     jax.debug.print("   [f5.debug] -> [{} ... {}]", A[0], A[-1])
    #     return A
    # print("   f5:", f5)

    # print("2. calling function (tracing and executing)...")
    # result = f5(jnp.array([1,2,3,4]))
    # print("   result:", *str(result).splitlines(), sep="\n    ")


def lesson6():
    print("== LESSON 6: ONLY SOME TYPES CAN BE STATIC/DYNAMIC =====")

    # print("1. defining and transforming function...")
    # @functools.partial(jax.jit, static_argnames=["n"])
    # def f6(
    #     n: int,
    #     a: Int[Array, "m"],
    # ) -> Int[Array, "n"]:
    #     print("   [f6.trace] n:", n)
    #     print("   [f6.trace] a:", a)
    #     jax.debug.print("   [f6.debug] n: {}", n)
    #     jax.debug.print("   [f6.debug] a: {}", a)
    #     return n + a
    # print("   f6:", f6)

    # print("2. calling function (tracing and executing)...")
    # result = f6(
    #     n=10,
    #     a=jnp.arange(5),
    # )
    # print("   result:", result)

    # print("3. calling function again (tracing and executing)...")
    # result = f6(
    #     n=11,
    #     a=jnp.arange(5),
    # )
    # print("   result:", result)
    # 
    # print("4. only hashable types can be passed as static params")
    # try:
    #     f6(
    #         n=jnp.array(1),
    #         a=jnp.arange(5),
    #     )
    # except Exception as e:
    #     print("   > "+"\n   > ".join(textwrap.wrap(str(e))))
    # 
    # print("5. only array(tree)s types can be passed as dynamic params")
    # try:
    #     f6(
    #         n=(min, max),
    #         a=(sum,),
    #     )
    # except Exception as e:
    #     print("   > "+"\n   > ".join(textwrap.wrap(str(e))))


def lesson7():
    print("== LESSON 7: EQUINOX AUTOMATIC FILTERING ===============")

    # print("1. defining function...")
    # def f7_nojit(
    #     n: int,
    #     a: Int[Array, "m"],
    # ) -> Int[Array, "n"]:
    #     print("   [f7.trace] n:", n)
    #     print("   [f7.trace] a:", a)
    #     # had to remove the debug calls for the builtin functions
    #     return n + a
    # print("   f7_nojit:", f7_nojit)

    # print("2. transforming function...")
    # f7_filterjit = eqx.filter_jit(f7_nojit)
    # print("   f7_filterjit:", f7_filterjit)

    # print("3. calling function with an int and an array...")
    # result = f7_filterjit(
    #     n=10,
    #     a=jnp.arange(5),
    # )
    # print("   result:", result)

    # print("4. calling function with an int and an array again...")
    # result = f7_filterjit(
    #     n=11,
    #     a=jnp.arange(5),
    # )
    # print("   result:", result)
    # 
    # print("5. calling function with two arrays...")
    # result = f7_filterjit(
    #     n=jnp.array(1),
    #     a=jnp.arange(5),
    # )
    # print("   result:", result)
    # 
    # print("6. calling function with two array trees...")
    # result = f7_filterjit(
    #     n=(min, max),
    #     a=(sum,),
    # )
    # print("   result:", result)
    # 
    # print("7. calling function with another two array trees...")
    # result = f7_filterjit(
    #     n=(sum,),
    #     a=(min, max),
    # )
    # print("   result:", result)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
