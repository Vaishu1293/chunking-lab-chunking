import math


def euclidean_distance(vector1, vector2):
    total = 0

    for value1, value2 in zip(vector1, vector2):
        total += (value1 - value2) ** 2

    return math.sqrt(total)


def dot_product(vector1, vector2):
    total = 0

    for value1, value2 in zip(vector1, vector2):
        total += value1 * value2

    return total


def vector_magnitude(vector):
    total = 0

    for value in vector:
        total += value ** 2

    return math.sqrt(total)


def cosine_similarity(vector1, vector2):

    numerator = dot_product(vector1, vector2)

    denominator = (
        vector_magnitude(vector1)
        * vector_magnitude(vector2)
    )

    return numerator / denominator