# "The token shall not be delimited by the end of the substitution."
foo=FOO; echo $(echo $foo)bar$(echo $foo)
