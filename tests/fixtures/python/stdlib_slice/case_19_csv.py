import csv


class Sink:
    def __init__(self):
        self.value = ""

    def write(self, value):
        self.value += value
        return len(value)


print(list(csv.reader(["name,note\n", 'Ada,"one, two"\n'])))
sink = Sink()
output = csv.writer(sink, lineterminator="\n")
output.writerow(["Ada", "one, two"])
print(sink.value)
