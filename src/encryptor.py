import string
import random

letters = list(string.ascii_uppercase)

def generate_derangement(letters):
    """Generate a derangement of the given letters (no letter remains in its original position)."""
    while True:
        deranged = random.sample(letters, len(letters))
        if all(l != d for l, d in zip(letters, deranged)):
            return deranged

def encrypt(text, mapping=None):
    """
    Encrypt text using a deranged substitution cipher.

    Parameters:
        text (str): Text to encrypt.
        mapping (list of str, optional): If provided, use this letter mapping.
                                        If None, generate a new deranged mapping.

    Returns:
        str: Encrypted text (cryptogram)
    """
    # Generate deranged mapping if none provided
    if mapping is None:
        mapping = generate_derangement(letters)

    result = ''
    for char in text:
        if char.isalpha():
            i = letters.index(char.upper())
            mapped = mapping[i]
            result += mapped if char.isupper() else mapped.lower()
        else:
            result += char
    return result
