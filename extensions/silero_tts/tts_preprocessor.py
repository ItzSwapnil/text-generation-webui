import re
import sys
import num2words

from num2words import num2words

PUNCTUATION = r'[\s,.?!/)\'\]>]'
ALPHABET_MAP = {
    "A": " Ei ",
    "B": " Bee ",
    "C": " See ",
    "D": " Dee ",
    "E": " Eee ",
    "F": " Eff ",
    "G": " Jee ",
    "H": " Eich ",
    "I": " Eye ",
    "J": " Jay ",
    "K": " Kay ",
    "L": " El ",
    "M": " Emm ",
    "N": " Enn ",
    "O": " Ohh ",
    "P": " Pee ",
    "Q": " Queue ",
    "R": " Are ",
    "S": " Ess ",
    "T": " Tee ",
    "U": " You ",
    "V": " Vee ",
    "W": " Double You ",
    "X": " Ex ",
    "Y": " Why ",
    "Z": " Zed "  # Zed is weird, as I (da3dsoul) am American, but most of the voice models sound British, so it matches
}


def preprocess(text: str) -> str:
    """
    Preprocesses a string by removing unnecessary characters, expanding numbers, and replacing abbreviations with their phonetic pronunciation.
    """
    text = remove_surrounded_chars(text)
    text = text.replace('"', '')
    text = text.replace('”', '').replace('“', '')  # right and left quote
    text = text.replace('‟', '').replace('„', '')  # right and left quote
    text = text.replace('\n', ' ')
    text = convert_num_locale(text)
    text = replace_negative(text)
    text = replace_roman(text)
    text = hyphen_range_to(text)
    text = num_to_words(text)
    text = replace_abbreviations(text)
    text = replace_lowercase_abbreviations(text)
    text = cleanup_whitespace(text)
    return text


def remove_surrounded_chars(text: str) -> str:
    """
    Removes characters surrounded by certain tags.
    """
    if re.search(r'(?<=alt=)(.*)(?=style=)', text, re.DOTALL):
        m = re.search(r'(?<=alt=)(.*)(?=style=)', text, re.DOTALL)
        text = m.group(0)
    text = re.sub(r'\*[^*]*?(\*|$)', '', text)
    return text


def convert_num_locale(text: str) -> str:
    """
    Converts numbers in the text to American format.
    """
    pattern = re.compile(r'(?:\s|^)\d{1,3}(?:\.\d{3})+(,\d+)(?:\s|$)')
    result = text
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start] + result[start:end].replace('.', '').replace(',', '.') + result[end:len(result)]

    # removes comma separators from existing American numbers
    pattern = re.compile(r'(\d),(\d)')
    result = pattern.sub(r'\1\2', result)

    return result


def replace_negative(text: str) -> str:
    """
    Replaces negative numbers with their phonetic pronunciation.
    """
    return re.sub(rf'(\s)(-)(\d+)({PUNCTUATION})', r'\1negative \3\4', text)


def replace_roman(text: str) -> str:
    """
    Replaces Roman numerals with their Arabic equivalent.
    """
    pattern = re.compile(rf'\s[IVXLCDM]{{2,}}{PUNCTUATION}')
    result = text
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start + 1] + str(roman_to_int(result[start + 1:end - 1])) + result[end - 1:len(result)]

    return result


def roman_to_int(s: str) -> int:
    """
    Converts a Roman numeral to its Arabic equivalent.
    """
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val


def hyphen_range_to(text: str) -> str:
    """
    Replaces hyphenated ranges with their phonetic pronunciation.
    """
    pattern = re.compile(r'(\d+
