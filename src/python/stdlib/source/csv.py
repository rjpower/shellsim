"""Small, deterministic subset of Python's csv module."""

QUOTE_MINIMAL = 0
QUOTE_ALL = 1
Error = ValueError


def _record_complete(text, quotechar):
    quoted = False
    index = 0
    while index < len(text):
        if text[index] == quotechar:
            if quoted and index + 1 < len(text) and text[index + 1] == quotechar:
                index += 1
            else:
                quoted = not quoted
        index += 1
    return not quoted


def _parse_record(text, delimiter, quotechar, skipinitialspace):
    if text == "":
        return []
    fields = []
    field = ""
    quoted = False
    index = 0
    while index < len(text):
        character = text[index]
        if quoted:
            if character == quotechar:
                if index + 1 < len(text) and text[index + 1] == quotechar:
                    field += quotechar
                    index += 1
                else:
                    quoted = False
            else:
                field += character
        elif character == quotechar and field == "":
            quoted = True
        elif character == delimiter:
            fields.append(field)
            field = ""
        elif not (skipinitialspace and character == " " and field == ""):
            field += character
        index += 1
    if quoted:
        raise Error("unexpected end of data")
    fields.append(field)
    return fields


class reader:
    def __init__(self, csvfile, delimiter=",", quotechar='"', skipinitialspace=False):
        self._rows = []
        pending = ""
        for line in csvfile:
            pending += line
            if _record_complete(pending, quotechar):
                self._rows.append(
                    _parse_record(pending.rstrip("\r\n"), delimiter, quotechar, skipinitialspace)
                )
                pending = ""
        if pending != "":
            raise Error("unexpected end of data")
        self.line_num = len(self._rows)
        self._position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._position >= len(self._rows):
            raise StopIteration
        row = self._rows[self._position]
        self._position += 1
        return row


def _quote_field(value, delimiter, quotechar, quoting):
    text = str(value)
    needs_quotes = (
        quoting == QUOTE_ALL
        or delimiter in text
        or quotechar in text
        or "\n" in text
        or "\r" in text
    )
    if quotechar in text:
        text = text.replace(quotechar, quotechar + quotechar)
    if needs_quotes:
        return quotechar + text + quotechar
    return text


class writer:
    def __init__(self, csvfile, delimiter=",", quotechar='"', lineterminator="\r\n", quoting=QUOTE_MINIMAL):
        self._file = csvfile
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.lineterminator = lineterminator
        self.quoting = quoting

    def writerow(self, row):
        fields = []
        for value in row:
            fields.append(_quote_field(value, self.delimiter, self.quotechar, self.quoting))
        return self._file.write(self.delimiter.join(fields) + self.lineterminator)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


class DictReader:
    def __init__(self, csvfile, fieldnames=None, delimiter=",", quotechar='"'):
        self._reader = reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        self.fieldnames = fieldnames
        if self.fieldnames is None:
            self.fieldnames = next(self._reader, None)

    def __iter__(self):
        return self

    def __next__(self):
        row = next(self._reader)
        while row == []:
            row = next(self._reader)
        result = {}
        for index in range(min(len(self.fieldnames), len(row))):
            result[self.fieldnames[index]] = row[index]
        return result


class DictWriter:
    def __init__(self, csvfile, fieldnames, delimiter=",", quotechar='"', lineterminator="\r\n"):
        self.fieldnames = fieldnames
        self._writer = writer(
            csvfile,
            delimiter=delimiter,
            quotechar=quotechar,
            lineterminator=lineterminator,
        )

    def writeheader(self):
        return self._writer.writerow(self.fieldnames)

    def writerow(self, rowdict):
        row = []
        for name in self.fieldnames:
            row.append(rowdict.get(name, ""))
        return self._writer.writerow(row)

    def writerows(self, rowdicts):
        for rowdict in rowdicts:
            self.writerow(rowdict)
