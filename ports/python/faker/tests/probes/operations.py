"""Exercise a bounded public API in the actual guest."""

from faker import Faker

first = Faker("en_US")
second = Faker("en_US")
first.seed_instance(17)
second.seed_instance(17)
assert first.name() == second.name()
assert "@" in first.email()
print("faker: guest public API probe passed")
