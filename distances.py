def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def norm_hamming_distance(x, y):
    if x == y:  # If the two tuples are the same, the distance is 0
        return 0
    else:
        # Otherwise, calculate the number of positions where the tuples differ
        diff_count = sum(a != b for a, b in zip(x, y))
        # Divide the count by the length of the tuples to get a value between 0 and 1
        return diff_count / len(x)
    
def jacard_similarity(x, y):
    # Check if both tuples are all zeros
    if all(a == 0 for a in x) and all(b == 0 for b in y):
        return 1.0
    else:
        # Calculate the intersection of the two sets
        intersection = sum(a and b for a, b in zip(x, y))
        # Calculate the union of the two sets
        union = sum(a or b for a, b in zip(x, y))
        # Calculate the Jaccard similarity as the ratio of intersection over union
        return intersection / union