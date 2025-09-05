def encrypt(text, new_letters):
    result = ''
    for char in text:
        if char.isalpha():
            i = letters.index(char.upper())
            result += new_letters[i]
        else:
            result += char
    return result
