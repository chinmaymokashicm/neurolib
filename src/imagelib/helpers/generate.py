import random

def generate_id(prefix: str, num_length: int, separator: str = "-") -> str:
    """
    Generate an ID with the given prefix and number length.
    """
    random_number: str = "".join([str(random.randint(0, 9)) for _ in range(num_length)])
    
    return f"{prefix}{separator}{random_number}"