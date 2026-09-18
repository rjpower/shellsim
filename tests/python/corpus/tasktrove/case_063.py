def outer():
    x = 1
    def inner():
        nonlocal x
        x += 1
        return x
    return inner
f = outer()
print(f(), f())
