# Python Interview Review

## Functions and Functional Programming

<details open>
<summary>What is a lambda function in Python?</summary>

A lambda function is an anonymous function defined using the `lambda` keyword.

**Syntax:** `lambda arguments: expression`

**Example:**
```python
add = lambda a, b: a + b
print(add(5, 3))
```
</details>

<details open>
<summary>What are Python decorators?</summary>

A decorator is a function that modifies the behavior of another function or class. It is commonly used to add functionality without changing the original code.

**Syntax:**
```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```
</details>

<details open>
<summary>What are *args and **kwargs in Python functions?</summary>

- `*args` allows a function to accept a variable number of positional arguments.
- `**kwargs` allows a function to accept a variable number of keyword arguments.

**Example:**
```python
def demo_function(*args, **kwargs):
    print("Args:", args)
    print("Kwargs:", kwargs)

demo_function(1, 2, 3, name="Alice", age=25)
```
</details>

## Object-Oriented Programming

<details open>
<summary>What is the purpose of the self keyword in Python?</summary>

The `self` keyword is used in instance methods within a class to refer to the current instance of the class. It allows you to access the instance’s attributes and methods.

**Example:**
```python
class MyClass:
    def __init__(self, value):
        self.value = value
    def print_value(self):
        print(self.value)
```
</details>

<details open>
<summary>What is the purpose of the __init__() function in Python?</summary>

The `__init__()` function is a special method in Python classes. It is used as a constructor to initialize a newly created object.

**Example:**
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person1 = Person("John", 30)
```
</details>

<details open>
<summary>How do you implement method overloading and overriding in Python?</summary>

- **Method Overloading:** Python does not support method overloading directly. It can be handled using default parameters.

**Example:**
```python
class Example:
    def show(self, x=None):
        if x is not None:
            print(f"Value: {x}")
        else:
            print("No argument passed")

obj = Example()
obj.show()      # No argument passed
obj.show(10)    # Value: 10
```

- **Method Overriding:** The child class overrides a method from the parent class.

**Example:**
```python
class Parent:
    def show(self):
        print("Parent class")

class Child(Parent):
    def show(self):
        print("Child class")

obj = Child()
obj.show()
```
</details>

<details open>
<summary>What is the new method in Python?</summary>

`__new__` is a special method used to create a new instance before `__init__` initializes it. It is useful when working with singleton classes.

**Example:**
```python
class Singleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

obj1 = Singleton()
obj2 = Singleton()
print(obj1 is obj2) # True (Only one instance created)
```
</details>

## Data Structures and Types

<details open>
<summary>What are Python’s data types?</summary>

- **Basic Data Types:** `int`, `float`, `str`, `bool`
- **Collections:** `list`, `tuple`, `set`, `dict`
- **Others:** `None`, `complex`
</details>

<details open>
<summary>What are Python modules and packages?</summary>

- **Module:** A single Python file containing functions, classes, and variables.
- **Package:** A collection of Python modules organized in directories, often with an `__init__.py` file to mark it as a package.

**Example:**
- `math` is a module.
- `NumPy` is a package that contains multiple modules.
</details>

## Copying and Memory Management

<details open>
<summary>Explain the difference between deepcopy() and copy() in Python.</summary>

- `copy()`: Creates a shallow copy of an object. It copies the reference to the nested objects.
- `deepcopy()`: Creates a deep copy, which means it recursively copies all objects, so changes to nested objects do not affect the original.

**Example:**
```python
import copy
a = [[1, 2], [3, 4]]

shallow = copy.copy(a)      # Shallow copy
deep = copy.deepcopy(a)     # Deep copy
```
</details>

<details open>
<summary>How do you manage memory in Python?</summary>

Python uses automatic garbage collection and reference counting to manage memory.

- **Reference Counting:** Python keeps track of how many references point to an object. When the reference count drops to zero, Python automatically deletes the object.
- **Garbage Collection (GC):** Python uses a garbage collector to remove circular references (objects referring to each other).
</details>

## Iterators, Generators, and Comprehensions

<details open>
<summary>What is a generator in Python?</summary>

A generator is a function that returns an iterator. Instead of returning all values at once, it yields values one by one using the `yield` keyword. It is memory-efficient for working with large datasets because it generates values on the fly.

**Example:**
```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for number in countdown(5):
    print(number)
```
</details>

<details open>
<summary>What are iterators in Python?</summary>

An iterator is an object that implements the `__iter__()` and `__next__()` methods. It allows you to traverse through all the elements of a collection, one by one.

**Example:**
```python
my_list = [1, 2, 3]
iterator = iter(my_list)
print(next(iterator))
```
</details>

<details open>
<summary>What are list comprehensions in Python?</summary>

List comprehensions provide a concise way to create lists.

**Syntax:** `[expression for item in iterable if condition]`

**Example:**
```python
squares = [x ** 2 for x in range(10)]
```
</details>

<details open>
<summary>What is the difference between a generator and a list comprehension?</summary>

- List comprehension creates the entire list in memory at once.
- Generators yield items one at a time using `yield`, making them more memory-efficient.

**Example:**
```python
list1 = [x**2 for x in range(5)]
print(list1)

gen = (x**2 for x in range(5))
print(next(gen))
print(next(gen))
```
</details>

## Operators and Exception Handling

<details open>
<summary>What is the difference between is and == in Python?</summary>

- `is`: Compares the identity of two objects (whether they are the same object in memory).
- `==`: Compares the values of two objects (whether they are equal).

**Example:**
```python
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True, values are equal
print(a is b)  # False, different objects in memory
```
</details>

<details open>
<summary>What is exception handling in Python?</summary>

Exception handling allows you to handle runtime errors gracefully using `try`, `except`, `else`, and `finally` blocks.

**Example:**
```python
try:
    x = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
else:
    print("No error")
finally:
    print("This will always execute")
```
</details>