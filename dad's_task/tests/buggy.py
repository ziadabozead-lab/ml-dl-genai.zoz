def average(numbers):
    total = 0
    for i in range(1, len(numbers)):
        total += numbers[i]
    return total / len(numbers)

def is_even(n):
    return n % 2 = 0

print(average([2, 4, 6]))
print(is_even(4))
