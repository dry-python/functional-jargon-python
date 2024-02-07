# Functional Programming Jargon

[![Build Status](https://github.com/dry-python/functional-jargon-python/workflows/test/badge.svg?event=push)](https://github.com/dry-python/functional-jargon-python/actions?query=workflow%3Atest)

Functional programming (FP) provides many advantages, and its popularity has been increasing as a result. However, each programming paradigm comes with its own unique jargon and FP is no exception. By providing a glossary, we hope to make learning FP easier.

This is a fork of [Functional Programming Jargon](https://github.com/jmesyou/functional-programming-jargon).

This document is WIP and pull requests are welcome!

__Table of Contents__
<!-- RM(noparent,notop) -->

* [Side effects](#side-effects)
* [Purity](#purity)
* [Idempotent](#idempotent)
* [Arity](#arity)
* [IO](#io)
* [Higher-Order Functions (HOF)](#higher-order-functions-hof)
* [Closure](#closure)
* [Partial Application](#partial-application)
* [Currying](#currying)
* [Function Composition](#function-composition)
* [Continuation](#continuation)
* [Point-Free Style](#point-free-style)
* [Predicate](#predicate)
* [Contracts](#contracts)
* [Category](#category)
* [Value](#value)
* [Constant](#constant)
* [Lift](#lift)
* [Referential Transparency](#referential-transparency)
* [Equational Reasoning](#equational-reasoning)
* [Lambda](#lambda)
* [Lambda Calculus](#lambda-calculus)
* [Lazy evaluation](#lazy-evaluation)
* [Functor](#functor)
* [Applicative Functor](#applicative-functor)
* [Monoid](#monoid)
* [Monad](#monad)
* [Comonad](#comonad)
* [Morphism](#morphism)
  * [Endomorphism](#endomorphism)
  * [Isomorphism](#isomorphism)
  * [Homomorphism](#homomorphism)
  * [Catamorphism](#catamorphism)
  * [Anamorphism](#anamorphism)
  * [Hylomorphism](#hylomorphism)
  * [Paramorphism](#paramorphism)
  * [Apomorphism](#apomorphism)
* [Setoid](#setoid)
* [Semigroup](#semigroup)
* [Foldable](#foldable)
* [Lens](#lens)
* [Type Signatures](#type-signatures)
* [Algebraic data type](#algebraic-data-type)
  * [Sum type](#sum-type)
  * [Product type](#product-type)
* [Option](#option)
* [Function](#function)
* [Partial function](#partial-function)


<!-- /RM -->


## Side effects

A function or expression is said to have a side effect if apart from returning a value, 
it interacts with (reads from or writes to) external mutable state:

```python
>>> print('This is a side effect!')
This is a side effect!
>>>
```

Or:

```python
>>> numbers = []
>>> numbers.append(1)  # mutates the `numbers` array
>>>
```


## Purity

A function is pure if the return value is only determined by its
input values, and does not produce any side effects.

This function is pure:

```python
>>> def add(first: int, second: int) -> int:
...    return first + second
>>>
```

As opposed to each of the following:

```python
>>> def add_and_log(first: int, second: int) -> int:
...    print('Sum is:', first + second)  # print is a side effect
...    return first + second
>>>
```


## Idempotent

A function is idempotent if reapplying it to its result does not produce a different result:

```python
>>> assert sorted([2, 1]) == [1, 2]
>>> assert sorted(sorted([2, 1])) == [1, 2]
>>> assert sorted(sorted(sorted([2, 1]))) == [1, 2]
>>>
```

Or:

```python
>>> assert abs(abs(abs(-1))) == abs(-1)
>>>
```


## Arity

The number of arguments a function takes. From words like unary, binary, ternary, etc. This word has the distinction of being composed of two suffixes, "-ary" and "-ity". Addition, for example, takes two arguments, and so it is defined as a binary function or a function with an arity of two. Such a function may sometimes be called "dyadic" by people who prefer Greek roots to Latin. Likewise, a function that takes a variable number of arguments is called "variadic," whereas a binary function must be given two and only two arguments, currying and partial application notwithstanding.

We can use the `inspect` module to know the arity of a function, see the example below:

```python
>>> from inspect import signature

>>> def multiply(number_one: int, number_two: int) -> int:  # arity 2
...     return number_one * number_two

>>> assert len(signature(multiply).parameters) == 2
>>>
```

### Arity Distinctions

#### Minimum Arity and Maximum Arity

The __minimum arity__ is the smallest number of arguments the function expects to work, the __maximum arity__ is the largest number of arguments function can take. Generally, these numbers are different when our function has default parameter values.

```python
>>> from inspect import getfullargspec
>>> from typing import Any

>>> def example(a: Any, b: Any, c: Any = None) -> None:  # mim arity: 2 | max arity: 3
...     pass

>>> example_args_spec = getfullargspec(example)
>>> max_arity = len(example_args_spec.args)
>>> min_arity = max_arity - len(example_args_spec.defaults)

>>> assert max_arity == 3
>>> assert min_arity == 2
>>>
```

#### Fixed Arity and Variable Arity

A function has __fixed arity__ when you have to call it with the same number of arguments as the number of its parameters and a function has __variable arity__ when you can call it with variable number of arguments, like functions with default parameters values.

```python
>>> from typing import Any

>>> def fixed_arity(a: Any, b: Any) -> None:  # we have to call with 2 arguments
...     pass

>>> def variable_arity(a: Any, b: Any = None) -> None:  # we can call with 1 or 2 arguments
...     pass
>>>
```

#### Definitive Arity and Indefinite Arity

When a function can receive a finite number of arguments it has __definitive arity__, otherwise if the function can receive an undefined number of arguments it has __indefinite arity__. We can reproduce the __indefinite arity__ using Python _*args_ and _**kwargs_, see the example below:

```python
>>> from typing import Any

>>> def definitive_arity(a: Any, b: Any = None) -> None: # we can call just with 1 or 2 arguments
...     pass

>>> def indefinite_arity(*args: Any, **kwargs: Any) -> None: # we can call with how many arguments we want
...     pass
>>>
```

### Arguments vs Parameters

There is a little difference between __arguments__ and __parameters__:

* __arguments__: are the values that are passed to a function
* __parameters__: are the variables in the function definition


## Higher-Order Functions (HOF)

A function that takes a function as an argument and/or returns a function, basically we can treat functions as a value.
In Python every function/method can be a Higher-Order Function.

The functions like `reduce`, `map` and `filter` are good examples of __HOF__, they receive a function as their first argument.
```python
>>> from functools import reduce

>>> reduce(lambda accumulator, number: accumulator + number, [1, 2, 3])
6
>>>
```

We can create our own __HOF__, see the example below:

```python
>>> from typing import Callable, TypeVar

>>> _ValueType = TypeVar('_ValueType')
>>> _ReturnType = TypeVar('_ReturnType')

>>> def get_transform_function() -> Callable[[str], int]:
...     return int

>>> def transform(
...     transform_function: Callable[[_ValueType], _ReturnType],
...     value_to_transform: _ValueType,
... ) -> _ReturnType:
...     return transform_function(value_to_transform)

>>> transform_function = get_transform_function()
>>> assert transform(transform_function, '42') == 42
>>>
```


## IO

IO basically means Input/Output, but it is widely used to just tell that a function is impure.

We have a special type (``IO``) and a decorator (``@impure``) to do that in Python:

```python
>>> import random
>>> from returns.io import IO, impure

>>> @impure
... def get_random_number() -> int:
...     return random.randint(0, 100)

>>> assert isinstance(get_random_number(), IO)
>>>
```

__Further reading__:
* [`IO` and `@impure` docs](https://returns.readthedocs.io/en/latest/pages/io.html)


## Closure (TODO)

A closure is a way of accessing a variable outside its scope.
Formally, a closure is a technique for implementing lexically scoped named binding. It is a way of storing a function with an environment.

A closure is a function that remembers the environment in which it was created. This means that it can access variables that were in scope at the time of the closure's creation, even after the block in which those variables were declared has finished executing.

```python
def make_adder(x):
    def adder(y):
        return x + y
    return adder

add_five = make_adder(5)
assert add_five(10) == 15
```

In the example above, `make_adder` returns a new function `adder` that takes a single argument `y`. The `adder` function adds `y` to `x`, where `x` is a parameter of the parent function `make_adder`. Even after `make_adder` has finished execution, `adder` remembers the value of `x` that was passed to `make_adder`.

Lexical scoping is the reason why it is able to find the values of x and add - the private variables of the parent which has finished executing. This value is called a Closure.

The stack along with the lexical scope of the function is stored in form of reference to the parent. This prevents the closure and the underlying variables from being garbage collected(since there is at least one live reference to it).

Lambda Vs Closure: A lambda is essentially a function that is defined inline rather than the standard method of declaring functions. Lambdas can frequently be passed around as objects.

A closure is a function that encloses its surrounding state by referencing fields external to its body. The enclosed state remains across invocations of the closure.

__Further reading/Sources__
* [Lambda Vs Closure](http://stackoverflow.com/questions/220658/what-is-the-difference-between-a-closure-and-a-lambda)
* [JavaScript Closures highly voted discussion](http://stackoverflow.com/questions/111102/how-do-javascript-closures-work)


## Partial Application

Partially applying a function means creating a new function by pre-filling some of the arguments to the original function.
You can also use `functools.partial` or `returns.curry.partial` to partially apply a function in Python:

```python
>>> from returns.curry import partial

>>> def takes_three_arguments(arg1: int, arg2: int, arg3: int) -> int:
...     return arg1 + arg2 + arg3

>>> assert partial(takes_three_arguments, 1, 2)(3) == 6
>>> assert partial(takes_three_arguments, 1)(2, 3) == 6
>>> assert partial(takes_three_arguments, 1, 2, 3)() == 6
>>>
```

The difference between `returns.curry.partial` and `functools.partial` 
is in how types are infered:

```python
import functools

reveal_type(functools.partial(takes_three_arguments, 1))
# Revealed type is 'functools.partial[builtins.int*]'

reveal_type(partial(takes_three_arguments, 1))
# Revealed type is 'def (arg2: builtins.int, arg3: builtins.int) -> builtins.int'
```

Partial application helps create simpler functions from more complex ones by baking in data when you have it. [Curried](#currying) functions are automatically partially applied.

__Further reading__
* [`@curry` docs](https://returns.readthedocs.io/en/latest/pages/curry.html#partial)
* [`functools` docs](https://docs.python.org/3/library/functools.html#functools.partial)


## Currying

The process of converting a function that takes multiple arguments into a function that takes them one at a time.

Each time the function is called it only accepts one argument and returns a function that takes one argument until all arguments are passed.

```python
>>> from returns.curry import curry

>>> @curry
... def takes_three_args(a: int, b: int, c: int) -> int:
...     return a + b + c

>>> assert takes_three_args(1)(2)(3) == 6
>>>
```

Some implementations of curried functions 
can also take several of arguments instead of just a single argument:

```python
>>> assert takes_three_args(1, 2)(3) == 6
>>> assert takes_three_args(1)(2, 3) == 6
>>> assert takes_three_args(1, 2, 3) == 6
>>>
```

Let's see what type `takes_three_args` has to get a better understanding of its features:

```python
reveal_type(takes_three_args)

# Revealed type is:
# Overload(
#   def (a: builtins.int) -> Overload(
#     def (b: builtins.int, c: builtins.int) -> builtins.int, 
#     def (b: builtins.int) -> def (c: builtins.int) -> builtins.int
#   ), 
#   def (a: builtins.int, b: builtins.int) -> def (c: builtins.int) -> builtins.int, 
#   def (a: builtins.int, b: builtins.int, c: builtins.int) -> builtins.int
# )'
```

__Further reading__
* [`@curry` docs](https://returns.readthedocs.io/en/latest/pages/curry.html#id3)
* [Favoring Curry](http://fr.umio.us/favoring-curry/)
* [Hey Underscore, You're Doing It Wrong!](https://www.youtube.com/watch?v=m3svKOdZijA)


## Function Composition

For example, you can compose `abs` and `int` functions like so:

```python
>>> assert abs(int('-1')) == 1
>>>
```

You can also create a third function 
that will have an input of the first one and an output of the second one:

```python
>>> from typing import Callable, TypeVar

>>> _FirstType = TypeVar('_FirstType')
>>> _SecondType = TypeVar('_SecondType')
>>> _ThirdType = TypeVar('_ThirdType')

>>> def compose(
...     first: Callable[[_FirstType], _SecondType],
...     second: Callable[[_SecondType], _ThirdType],
... ) -> Callable[[_FirstType], _ThirdType]:
...     return lambda argument: second(first(argument))

>>> assert compose(int, abs)('-1') == 1
>>>
```

We already have this functions defined as `returns.functions.compose`!

```python
>>> from returns.functions import compose
>>> assert compose(bool, str)([]) == 'False'
>>>
```

__Further reading__
* [`compose` docs](https://returns.readthedocs.io/en/latest/pages/functions.html#compose)


## Continuation

A continuation represents the future of a computation. It is a way of structuring some parts of a program so that the execution can be paused and resumed. Continuations are a deep concept that can be used to implement advanced control structures, such as coroutines, exception handling, or backtracking.

In Python, continuations can be implemented using generators:

```python
def print_numbers():
    for i in range(5):
        print(i)
        yield

gen = print_numbers()  # Create a generator
next(gen)  # Prints 0 and pauses
next(gen)  # Resumes and prints 1
```

In this example, the `yield` statement is used to pause the execution of `print_numbers`. Each call to `next` on the generator resumes where the function left off.

Continuations are often seen in asynchronous programming when the program needs to wait to receive data before it can continue. The response is often passed off to the rest of the program, which is the continuation, once it's been received.

```python
import asyncio

async def fetch_data():
    print("Start fetching")
    await asyncio.sleep(2)  # Simulate an I/O operation e.g., fetching data from a database
    print("Done fetching")
    return {'data': 123}

async def main_continuation():
    # The continuation is here after the data is fetched
    data = await fetch_data()
    # The rest of the code is the continuation of the fetch_data function
    print(f"Received data: {data}")

# Run the main_continuation coroutine
asyncio.run(main_continuation())
```

In this example, `fetch_data` is an asynchronous function that simulates a delay (such as a network request) with `await asyncio.sleep(2)`. The `await` keyword is used to wait for the completion of `fetch_data` without blocking the entire program. Once `fetch_data` is complete, the execution continues with the next line, which is effectively the continuation of the `fetch_data` operation.

The `main_continuation` function awaits the result of `fetch_data` and then processes the data, which is the continuation after the asynchronous operation has completed. The `asyncio.run(main_continuation())` call at the bottom is used to run the main coroutine, which in turn waits for the `fetch_data` coroutine.


## Point-Free Style

Point-Free is a style of writting code without using any intermediate variables.

Basically, you will end up with long chains of direct function calls.
This style usually requires [currying](#currying) or other [Higher-Order functions](#higher-order-functions-hof). 
This technique is also sometimes called "Tacit programming".

The most common example of Point-Free programming style is Unix with pipes:

```bash
ps aux | grep [k]de | gawk '{ print $2 }'
```

It also works for Python, let's say you have this function composition:

```python
>>> str(bool(abs(-1)))
'True'
>>>
```

It might be problematic method methods on the first sight, because you need an instance to call a method on.
But, you can always use HOF to fix that and compose normally:

```python
>>> from returns.pipeline import flow
>>> from returns.pointfree import map_
>>> from returns.result import Success

>>> assert flow(
...     Success(-2),
...     map_(abs),
...     map_(range),
...     map_(list),
... ) == Success([0, 1])
>>>
```

__Further reading:__
* [Pointfree docs](https://returns.readthedocs.io/en/latest/pages/pointfree.html)


## Predicate

A predicate is a function that returns true or false for a given value.
So, basically a predicate is an alias for `Callable[[_ValueType], bool]`.

It is very useful when working with `if`, `all`, `any`, etc.

```python
>>> def is_long(item: str) -> bool:
...     return len(item) > 3

>>> assert all(is_long(item) for item in ['1234', 'abcd'])
>>>
```

__Futher reading__
* [Predicate logic](https://en.wikipedia.org/wiki/Predicate_functor_logic)
* [`cond` docs](https://returns.readthedocs.io/en/latest/pages/pointfree.html#cond)


## Contracts

A contract in programming specifies the obligations and guarantees of the behavior from a function or expression at runtime. This can be implemented in Python using assertions that serve as runtime checks.

```python
def add_positive_numbers(x, y):
    assert x > 0 and y > 0, "Both numbers must be positive"
    return x + y

add_positive_numbers(1, 1)  # Works fine
add_positive_numbers(-1, 1)  # Raises an AssertionError
```

In the example above, the contract is that both `x` and `y` must be positive numbers. The `assert` statement checks this contract at runtime.

## Category

In programming, a category consists of objects and morphisms (functions) between these objects. Categories must satisfy three properties:

1. Composition: Morphisms can be composed, and the composition is associative.
2. Identity: Each object has an identity morphism that acts as a no-op.
3. Closure: The composition of two morphisms is also a morphism in the category.

Since these rules govern composition at very abstract level, category theory is great at uncovering new ways of composing things.

In Python, we can think of types as objects and functions as morphisms:

```python
def identity(x):
    return x

def compose(f, g):
    return lambda x: f(g(x))

# The identity function acts as a no-op morphism.
assert identity(5) == 5

# The composition of two functions is also a function.
f = lambda x: x + 1
g = lambda x: x * 2
assert compose(f, g)(5) == 11
```

__Further reading__

* [Category Theory for Programmers](https://bartoszmilewski.com/2014/10/28/category-theory-for-programmers-the-preface/)

## Value

In programming, a value is anything that can be assigned to a variable or passed to a function as an argument. In Python, values can be of various types, such as numbers, strings, lists, or even functions.

```python
x = 42  # 42 is a value
y = "hello"  # "hello" is a value
z = [1, 2, 3]  # [1, 2, 3] is a value
```

## Constant

A constant is a value that cannot be altered by the program during normal execution. In Python, constants are usually defined at the module level and written in all capital letters:

```python
PI = 3.14159

def circumference(radius):
    return 2 * PI * radius

# PI is a constant, its value should not be changed.
```

## Lift

Lifting is a concept in functional programming where you take a function that operates on values and transform it into a function that operates on values inside a context (like a container or a monad).

For example, if you have a function `add` that adds two numbers, you can lift this function to operate on lists of numbers:

```python
def add(x, y):
    return x + y

def lift_to_list(f):
    return lambda x, y: [f(a, b) for a, b in zip(x, y)]

add_lists = lift_to_list(add)
assert add_lists([1, 2], [3, 4]) == [4, 6]
```

In this example, `lift_to_list` takes a function `f` and returns a new function that applies `f` to corresponding elements of two lists.

## Referential Transparency

An expression is referentially transparent if it can be replaced with its corresponding value without changing the program's behavior. This concept is a key feature of pure functions in functional programming.

```python
def square(x):
    return x * x

# The function call square(2) is referentially transparent,
# because it can be replaced with its result, 4, without changing the meaning.
assert square(2) == 4
```

In the example above, `square(2)` is referentially transparent because we can replace it with `4` anywhere in our code without affecting the outcome.

## Equational Reasoning

Equational reasoning is a technique used in functional programming to reason about program behavior. It relies on the substitution of expressions with their equivalent values, which is possible due to referential transparency.

```python
def add(a, b):
    return a + b

def square(x):
    return x * x

# We can reason that:
# square(add(1, 2)) == square(3) == 9
# This is possible because both add and square are pure functions.
assert square(add(1, 2)) == 9
```

In this example, we can substitute `add(1, 2)` with `3` and `square(3)` with `9`, allowing us to reason about the code's behavior in a straightforward way.

## Lambda

A lambda is an anonymous function that can be defined inline and passed around as a value.

```python
# A lambda function that adds 1 to its argument
add_one = lambda x: x + 1

# Lambdas are often used in higher-order functions
numbers = [1, 2, 3]
incremented_numbers = list(map(lambda x: x + 1, numbers))
assert incremented_numbers == [2, 3, 4]
```

In the example above, `lambda x: x + 1` is a lambda function that increments its input by 1.

## Lambda Calculus

Lambda calculus is a formal system in mathematical logic for expressing computation based on function abstraction and application using variable binding and substitution. It is a universal model of computation that can be used to simulate any Turing machine.

```python
# Lambda calculus is not directly implemented in Python,
# but Python's functions and lambdas are inspired by it.
```

## Lazy evaluation

Lazy evaluation is a strategy that delays the evaluation of an expression until its value is needed. It can improve performance by avoiding unnecessary calculations, and it can create potential for infinite data structures.

```python
def lazy_add(a, b):
    def compute():
        return a + b
    return compute

# The sum is not computed until we call the returned function.
sum_function = lazy_add(1, 2)
assert sum_function() == 3
```

In this example, `lazy_add` returns a function that, when called, computes the sum of `a` and `b`. The sum is not computed until `compute` is called.

## Functor

An object that implements a `map` method which, while running over each value in the object to produce a new object, adheres to two rules:

__Identity law__

```python
functor.map(lambda x: x) == functor
```

__Associative law__

```python
functor.map(compose(f, g)) == functor.map(g).map(f)
```

Sometimes `Functor` can be called `Mappable` to its `.map` method.
You can have a look at the real-life [`Functor` interface](https://github.com/dry-python/returns/blob/master/returns/interfaces/mappable.py):

```python
>>> from typing import Callable, TypeVar
>>> from returns.interfaces.mappable import Mappable1 as Functor
>>> from returns.primitives.hkt import SupportsKind1

>>> _FirstType = TypeVar('_FirstType')
>>> _NewFirstType = TypeVar('_NewFirstType')

>>> class Box(SupportsKind1['Box', _FirstType], Functor[_FirstType]):
...     def __init__(self, inner_value: _FirstType) -> None:
...         self._inner_value = inner_value
...
...     def map(
...         self,
...         function: Callable[[_FirstType], _NewFirstType],
...     ) -> 'Box[_NewFirstType]':
...         return Box(function(self._inner_value))
...
...     def __eq__(self, other) -> bool:
...         return type(other) == type(self) and self._inner_value == other._inner_value

>>> assert Box(-5).map(abs) == Box(5)
>>>
```

__Further reading:__

- [Functor interface docs](https://returns.readthedocs.io/en/latest/pages/interfaces.html#mappable)


## Applicative Functor

An Applicative Functor is an object with `apply` and `.from_value` methods:
- `.apply` applies a function in the object to a value in another object of the same type. Somethimes this method is also called `ap`
- `.from_value` creates a new Applicative Functor from a pure value. Sometimes this method is also called `pure`

All Applicative Functors must also follow [a bunch of laws](https://returns.readthedocs.io/en/latest/pages/interfaces.html#applicative).

__Further reading:__

- [`Applicative Functor` interface docs](https://github.com/dry-python/returns/blob/master/returns/interfaces/applicative.py)


## Monoid

An object with a function that "combines" that object with another of the same type
and an "empty" value, which can be added with no effect.

One simple monoid is the addition of numbers 
(with `__add__` as an addition function and `0` as an empty element):

```python
>>> assert 1 + 1 + 0 == 2
>>>
```

Tuples, lists, and strings are also monoids:

```python
>>> assert (1,) + (2,) + () == (1, 2)
>>> assert [1] + [2] + [] == [1, 2]
>>> assert 'a' + 'b' + '' == 'ab'
>>>
```

## Monad

A monad is a design pattern used to handle program-wide concerns (like state or I/O) in a functional way. It's an object with `bind` and `return` methods that follow certain laws.

```python
class Monad:
    def __init__(self, value):
        self.value = value

    def bind(self, func):
        return func(self.value)

    def return_(self):
        return self

# Example usage:
# Let's say we have a monad instance `m` and a function `f` that returns a monad.
# We can apply `f` to the value wrapped by `m` using `bind`:
# m.bind(f)
```

In Python, monads are not as prevalent as in languages like Haskell, but the concept can still be applied.

## Comonad

A comonad is conceptually the dual of a monad. It provides a way to extract a value from a context and extend a computation across a context.

```python
class Comonad:
    def __init__(self, value):
        self.value = value

    def extract(self):
        return self.value

    def extend(self, func):
        return Comonad(func(self))

# Example usage:
# Given a comonad `w` and a function `f` that takes a comonad and returns a value,
# we can create a new comonad with the result of applying `f` to `w` using `extend`:
# w.extend(f)
```

Comonads are less common in everyday programming but can be useful in certain contexts like functional reactive programming.

## Morphism

### Endomorphism

An endomorphism is a function where the input type is the same as the output type.

```python
# An endomorphism that converts a string to uppercase
def to_uppercase(s: str) -> str:
    return s.upper()

# An endomorphism that decrements a number
def decrement(x: int) -> int:
    return x - 1
```

### Isomorphism

An isomorphism consists of two functions that are inverses of each other. They allow for lossless conversion between two types.

```python
def to_str(n: int) -> str:
    return str(n)

def to_int(s: str) -> int:
    return int(s)

# to_str and to_int are isomorphisms if we ignore the fact that not all strings can be converted to integers.
assert to_int(to_str(42)) == 42
assert to_str(to_int("42")) == "42"
```

### Homomorphism

A homomorphism is a structure-preserving map between two algebraic structures.

```python
# A homomorphism between (integers under addition) and (strings under concatenation)
def add_to_str(x: int, y: int) -> str:
    return str(x + y)

assert add_to_str(1, 2) == "3"
```

### Catamorphism

A catamorphism is a way to deconstruct data structures into a single value.

```python
from functools import reduce

# A catamorphism that sums a list of numbers
def sum_list(numbers: list) -> int:
    return reduce(lambda acc, x: acc + x, numbers, 0)

assert sum_list([1, 2, 3]) == 6
```

### Anamorphism

An anamorphism is a way to construct data structures from a single value.

```python
def unfold(predicate, function, seed):
    result = []
    while predicate(seed):
        value, seed = function(seed)
        result.append(value)
    return result

# An anamorphism that generates a range of numbers
def range_anamorphism(start, stop):
    return unfold(lambda x: x < stop, lambda x: (x, x + 1), start)

assert range_anamorphism(0, 5) == [0, 1, 2, 3, 4]
```

### Hylomorphism

A hylomorphism is a combination of an anamorphism and a catamorphism.

```python
# A hylomorphism that constructs a range of numbers and then sums them
def hylo_range_sum(start, stop):
    return sum_list(range_anamorphism(start, stop))

assert hylo_range_sum(0, 5) == 10
```

### Paramorphism

A paramorphism is a recursive function that has access to the results of the recursive call and the original data structure.

```python
def para(numbers, acc=0):
    if not numbers:
        return acc
    head, *tail = numbers
    return para(tail, acc + head)

# A paramorphism that sums a list of numbers
assert para([1, 2, 3, 4]) == 10
```

### Apomorphism

An apomorphism is a dual to paramorphism, allowing for early termination in an unfold.

```python
def apo(predicate, function, seed):
    result = []
    while predicate(seed):
        value, seed, done = function(seed)
        result.append(value)
        if done:
            break
    return result

# An apomorphism that generates a range of numbers but can stop early
def range_apomorphism(start, stop, early_stop):
    return apo(
        lambda x: x < stop,
        lambda x: (x, x + 1, x >= early_stop),
        start
    )

assert range_apomorphism(0, 10, 5) == [0, 1, 2, 3, 4]
```

## Setoid

A setoid is an object that has an `equals` method which can be used to compare other objects of the same type for equality.

```python
class Setoid:
    def __init__(self, value):
        self.value = value

    def equals(self, other):
        return self.value == other.value

# Example usage:
a = Setoid(1)
b = Setoid(1)
c = Setoid(2)

assert a.equals(b)
assert not a.equals(c)
```

## Semigroup

A semigroup is an algebraic structure with a `concat` method that combines it with another object of the same type.

```python
class Semigroup:
    def __init__(self, value):
        self.value = value

    def concat(self, other):
        return Semigroup(self.value + other.value)

# Example usage:
a = Semigroup("Hello, ")
b = Semigroup("World!")

assert a.concat(b).value == "Hello, World!"
```

## Foldable

A foldable is an object that has a `reduce` method that applies a function against an accumulator and each element in the object to reduce it to a single value.

```python
class Foldable(list):
    def reduce(self, function, initial):
        accumulator = initial
        for value in self:
            accumulator = function(accumulator, value)
        return accumulator

# Example usage:
numbers = Foldable([1, 2, 3, 4])
sum_of_numbers = numbers.reduce(lambda acc, x: acc + x, 0)

assert sum_of_numbers == 10
```

## Lens

A lens is a composable structure that pairs a getter and a setter for an immutable update of a data structure.

```python
class Lens:
    def __init__(self, getter, setter):
        self._getter = getter
        self._setter = setter

    def view(self, data):
        return self._getter(data)

    def set(self, data, value):
        return self._setter(data, value)

    def over(self, data, function):
        return self.set(data, function(self.view(data)))

# Example usage:
name_lens = Lens(lambda data: data['name'], lambda data, value: {**data, 'name': value})
person = {'name': 'Alice', 'age': 25}

assert name_lens.view(person) == 'Alice'
assert name_lens.set(person, 'Bob') == {'name': 'Bob', 'age': 25}
```

Lenses can be composed to focus on nested data structures.

```python
# Composing lenses to focus on nested structures
address_lens = Lens(lambda data: data['address'], lambda data, value: {**data, 'address': value})
street_lens = Lens(lambda data: data['street'], lambda data, value: {**data, 'street': value})

full_address_lens = address_lens.compose(street_lens)
person_with_address = {'name': 'Alice', 'address': {'street': '123 Main St', 'city': 'Anytown'}}

assert full_address_lens.view(person_with_address) == '123 Main St'
```

## Type Signatures

Type signatures are annotations that specify the types of inputs and outputs for functions. They help in understanding how a function can be used and are essential for static type checking.

```python
from typing import List, Callable

# Type signature for a function that takes an integer and returns a string
def int_to_str(x: int) -> str:
    return str(x)

# Type signature for a higher-order function that takes a function as an argument
def apply_function(f: Callable[[int], str], x: int) -> str:
    return f(x)

# Type signature for a function that takes a list of integers and returns a list of strings
def list_int_to_str(numbers: List[int]) -> List[str]:
    return [str(number) for number in numbers]
```

## Algebraic data type

An algebraic data type (ADT) is a type formed by combining other types. Two common kinds of algebraic types are sum types and product types.

### Sum type

A sum type is a type that can hold a value that could be one of several different types. It is called a sum type because the number of possible values it can represent is the sum of the possible values of its variants.

```python
from typing import Union

# A sum type representing either an integer or a string
NumberOrString = Union[int, str]

def handle_value(value: NumberOrString):
    if isinstance(value, int):
        return f"Integer: {value}"
    elif isinstance(value, str):
        return f"String: {value}"

assert handle_value(42) == "Integer: 42"
assert handle_value("hello") == "String: hello"
```

### Product type

A product type is a type that combines several values into one compound value. It is called a product type because the number of possible values it can represent is the product of the possible values of its components.

```python
from typing import NamedTuple

# A product type representing a point in 2D space
class Point(NamedTuple):
    x: int
    y: int

point = Point(1, 2)
assert point.x == 1
assert point.y == 2
```

## Option

the `Option` type is a powerful construct used to represent an optional value: a value that might exist or might not. It's a way to avoid `null` references and the errors they can cause, known as "null pointer exceptions."

The `Option` type is a sum type, meaning it can be one of several variants. In the case of `Option`, there are typically two variants:

1. `Some`: Indicates the presence of a value. The `Some` variant wraps an actual value.
2. `None`: Indicates the absence of a value. There is no value present.

This pattern forces the programmer to explicitly handle the case where a value might be missing, leading to safer and more predictable code.

In languages like Haskell, this type is known as `Maybe`, with the variants being `Just` (for `Some`) and `Nothing` (for `None`). In Rust, it's called `Option`, with the variants `Some` and `None`.

```python
from typing import Generic, TypeVar, Union

T = TypeVar('T')

class Some(Generic[T]):
    def __init__(self, value: T):
        self.value = value

class NoneType:
    pass

NoneValue = NoneType()

Option = Union[Some[T], NoneType]

def divide(dividend: int, divisor: int) -> Option[int]:
    if divisor == 0:
        return NoneValue
    else:
        return Some(dividend // divisor)
```

The example above is a Pythonic implementation of the `Option` type, using classes to represent the `Some` variant and Python's built-in `None` to represent the absence of a value.

Here's a breakdown of the example:

- `Some` class: This generic class is used to wrap a value of any type `T`. It provides an `__init__` method to set the value, an `__eq__` method to compare two `Some` instances, and a `__repr__` method for a string representation that is helpful for debugging.

- `divide` function: This function attempts to perform integer division. It returns an `Optional[Some[int]]`, which means the return type could be an instance of `Some` containing an integer or `None`. If the divisor is zero, the function returns `None` to signify that the division cannot be performed. Otherwise, it returns a `Some` object containing the result of the division.

- Example usage: The code demonstrates how to use the `divide` function and how to handle its return value. It checks if the result is `None` to determine if the division was successful. If the result is not `None`, it accesses the value inside the `Some` object using `.value`.

This implementation encourages explicit handling of the case where division by zero might occur, thus avoiding the potential for runtime errors that could arise from an unhandled division by zero. It's a simple yet effective way to introduce some of the safety and expressiveness of functional programming into Python.

## Function

A __function__ `f: A -> B` is an abstraction that encapsulates a computation or transformation: it takes an input of type `A` and produces an output of type `B`. The function guarantees that for every input of type `A`, there will be a corresponding output of type `B`, and this output is solely determined by the input value. This property is known as referential transparency, which means that the function can be replaced with its corresponding output value without changing the program's behavior. Functions in programming are typically pure, meaning they do not cause any side effects outside of producing a return value. This makes functions predictable and easy to reason about.

Here is an example of a simple function in Python:

```python
def add_one(x: int) -> int:
    return x + 1

# Usage
result = add_one(3)  # result is 4
```

## Partial Function

A partial function is a function that is not defined for all possible inputs of the specified type. In other words, it only provides an output for a subset of possible inputs. When a partial function is given an input that it does not handle, it may throw an exception, return an unexpected result, or never terminate.

Here are some examples of partial functions:

```python
def inverse(x: float) -> float:
    if x == 0:
        raise ValueError("Cannot divide by zero")
    return 1 / x

def get_first_element(lst: list) -> int:
    return lst[0]  # This will fail if the list is empty
```

### Dealing with Partial Functions

To mitigate the risks associated with partial functions, we can convert them into total functions. A total function is defined for all possible inputs and always produces a result. We can handle the cases where the partial function is not defined by providing default values, raising exceptions, or using constructs like `Option` to represent the possibility of failure.

Here's how we might handle the partial functions above in a safer way:

```python
from typing import Optional

def safe_inverse(x: float) -> Optional[float]:
    if x == 0:
        return None
    return 1 / x

def safe_get_first_element(lst: list) -> Optional[int]:
    if not lst:
        return None
    return lst[0]

# Usage
result_inverse = safe_inverse(0)  # result_inverse is None
result_first_element = safe_get_first_element([])  # result_first_element is None
```

In these examples, we return `None` when the function would otherwise be undefined. This approach forces the caller to handle the possibility that there may not be a valid return value, making the code more robust and predictable.
