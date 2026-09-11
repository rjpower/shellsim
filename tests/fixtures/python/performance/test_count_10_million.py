def test_count_to_ten_million():
    count = 0
    while count < 10_000_000:
        count += 1
    assert count == 10_000_000
