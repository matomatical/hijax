"""
jit dojo: lessons on just-in-time compilation in JAX.

Lessons:

0. Start with a plain python function using jax.numpy operations.
1. Use `jax.jit` to compile the function, see what happens.
2. The different phases of jit: tracing, compilation, and execution.
3. Mark certain parameters as 'static' to make them available at trace time.

Future lessons:

4. shapes of dynamic arguments are statically available
5. branching and looping happen at trace time
"""

from typing import Literal
import jax
import jax.numpy as jnp
from jaxtyping import Int, Array


def main(lesson: Literal[0,1,2,3]):
    match lesson:
        case 0: lesson0()
        case 1: lesson1()
        case 2: lesson2()
        case 3: lesson3()


def lesson0():
    print("== LESSON 0: PLAIN JAX FUNCTION ========================")

    print("1. defining function...")
    def f0_nojit(
        a: Int[Array, "n"],
        b: Int[Array, "n"],
    ) -> Int[Array, "n"]:
        print(" [f0] a:", a)
        print(" [f0] b:", b)
        c = a + b
        print(" [f0] ->", c)
        return c
    print(" f0_nojit:", f0_nojit)

    print("2. calling function...")
    result = f0_nojit(jnp.arange(8), jnp.ones(8, dtype=int))
    print(" result:", result)


def lesson1():
    print("== LESSON 1: JUST-IN-TIME COMPILATION ==================")

    print("1. defining function...")
    def f1_nojit(
        a: Int[Array, "n"],
        b: Int[Array, "n"],
    ) -> Int[Array, "n"]:
        print(" [f1] a:", a)
        print(" [f1] b:", b)
        c = a + b
        print(" [f1] ->", c)
        return c
    print(" f1_nojit:", f1_nojit)

    print("2. transforming function...")
    f1_jit = jax.jit(f1_nojit)
    print(" f1_jit:", f1_jit)


    print("3. calling function...")
    result = f1_jit(jnp.arange(8), jnp.ones(8, dtype=int))
    print(" result:", result)


def lesson2():
    print("== LESSON 2: DISTINGUISHING TRACING VS. EXECUTION ======")

    print("1. defining and transforming function...")
    @jax.jit
    def f2(
        a: Int[Array, "n"],
        b: Int[Array, "n"],
    ) -> Int[Array, "n"]:
        print(" [f2.trace] a:", a)
        jax.debug.print(" [f2.debug] a: {}", a)
        print(" [f2.trace] b:", b)
        jax.debug.print(" [f2.debug] b: {}", b)
        c = a + b
        print(" [f2.trace] ->", c)
        jax.debug.print(" [f2.debug] -> {}", c)
        return c
    print(" f2:", f2)

    print("2. calling function (tracing and executing)...")
    result = f2(jnp.arange(8), jnp.ones(8, dtype=int))
    print(" result:", result)

    print("3. calling function with same types (only executing)...")
    result = f2(jnp.arange(8), jnp.ones(8, dtype=int))
    print(" result:", result)

    print("4. calling function with different types (tracing and executing)...")
    result = f2(jnp.arange(4), jnp.ones(4, dtype=int))
    print(" result:", result)


def lesson3():
    print("== LESSON 3: STATIC VS. DYNAMIC ARGUMENTS ==============")

    print("1. defining function...")
    def f3_nojit(n: int) -> Int[Array, "n n"]:
        print(" [f3.trace] n:", n)
        jax.debug.print(" [f3.debug] n: {}", n)
        a = jnp.arange(n) + 1
        print(" [f3.trace] a:", a)
        jax.debug.print(" [f3.debug] a: {}", a)
        b = jnp.diag(a)
        print(" [f3.trace] ->", b)
        jax.debug.print(" [f3.debug] -> [{} ... {}]", b[0], b[-1])
        return b
    print(" f3_nojit:", f3_nojit)

    print("2. transforming function...")
    f3_jit = jax.jit(f3_nojit, static_argnames=['n'])
    # NOTE: without 'static_argnames' it raises an error
    print(" f3_jit:", f3_jit)

    print("3. calling function (tracing and executing)...")
    result = f3_jit(4)
    print(" result:", *str(result).splitlines(), sep="\n  ")


    print("4. calling function again (tracing and executing)...")
    result = f3_jit(3)
    print(" result:", *str(result).splitlines(), sep="\n  ")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
