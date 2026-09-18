def make_adder(x):
    def add(y):
        return x + y
    return add
print(make_adder(4)(3))
