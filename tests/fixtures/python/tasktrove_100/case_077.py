def classify(n):
    if n < 0:
        return "negative"
    return "nonnegative"
print(classify(-1), classify(2))
