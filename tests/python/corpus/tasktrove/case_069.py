class C:
    def __enter__(self):
        return 7
    def __exit__(self, a, b, c):
        return False
with C() as value:
    print(value)
