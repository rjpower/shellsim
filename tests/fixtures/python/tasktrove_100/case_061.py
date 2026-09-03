class Box:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
b = Box(7)
print(b.get(), b.value)
