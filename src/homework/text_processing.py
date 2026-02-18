
__author__ = 'Anushka Basu'

import re
from datetime import date

TITLE = 'title'
YEAR = 'year'
CHAPTERS = 'chapters'
NUMBER = 'number'
TOKEN_COUNT = 'token_count'

def chronicles_of_narnia(filepath: str) -> dict:
    """
    Extract structured information from Chronicles of Narnia text file.

    :param filepath: Path to the input text file.
    :return: Dictionary containing book metadata and chapter statistics.
    """

    # Book title line looks like: The Lion , the Witch and the Wardrobe ( 1950 )
    re_book = re.compile(r'^(.+?)\s+\(\s+(\d{4})\s+\)\s*$')

    # Chapter heading line looks like: CHAPTER I  OR  Chapter IX
    re_chapter = re.compile(r'^(?:CHAPTER|Chapter)\s+([IVXLCDM]+)\s*$')

    def roman_to_int(roman: str) -> int:
        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total, prev = 0, 0
        for ch in reversed(roman):
            val = roman_map[ch]
            if val < prev:
                total -= val
            else:
                total += val
            prev = val
        return total

    narnia_data: dict = {}

    current_book_title = None
    current_book_year = None
    current_chapters = []

    current_chapter = None
    current_token_count = 0
    counting = False

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()


        # New book starts
        m_book = re_book.fullmatch(stripped)
        if m_book:
            # close out previous chapter
            if current_chapter is not None:
                current_chapter[TOKEN_COUNT] = current_token_count
                current_chapters.append(current_chapter)
                current_chapter = None
                current_token_count = 0
                counting = False

            # close out previous book
            if current_book_title is not None:
                current_chapters.sort(key=lambda d: d[NUMBER])
                narnia_data[current_book_title] = {
                    TITLE: current_book_title,
                    YEAR: current_book_year,
                    CHAPTERS: current_chapters
                }

            # start new book
            current_book_title = m_book.group(1)   # preserves spacing from text
            current_book_year = int(m_book.group(2))
            current_chapters = []

            i += 1
            continue

        # New chapter starts (only if we're inside a book)
        m_ch = re_chapter.match(stripped)
        if m_ch and current_book_title is not None:
            # close out previous chapter
            if current_chapter is not None:
                current_chapter[TOKEN_COUNT] = current_token_count
                current_chapters.append(current_chapter)

            chap_num = roman_to_int(m_ch.group(1))

            # Next non-empty line is the chapter title
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1

            chap_title = lines[j].strip() if j < len(lines) else ""

            current_chapter = {
                NUMBER: chap_num,
                TITLE: chap_title
            }
            current_token_count = 0
            counting = True

            i = j + 1  # IMPORTANT: skip the title line so we do NOT count it
            continue

        # Count tokens for chapter body (tokens are whitespace-separated already)
        if counting and current_chapter is not None:
            if stripped:
                current_token_count += len(stripped.split())

        i += 1

    # finalize last chapter
    if current_chapter is not None:
        current_chapter[TOKEN_COUNT] = current_token_count
        current_chapters.append(current_chapter)

    # finalize last book
    if current_book_title is not None:
        current_chapters.sort(key=lambda d: d[NUMBER])
        narnia_data[current_book_title] = {
            TITLE: current_book_title,
            YEAR: current_book_year,
            CHAPTERS: current_chapters
        }

    return narnia_data

EMAIL = 'email'
DATE = 'date'
URL = 'url'
CITE = 'cite'


def regular_expressions(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    if not s:
        return None
    """
    Identifies the type of a given text pattern.

    :param text: String to classify.
    :return: One of "email", "date", "url", "cite"; None, if no pattern matches.
    """
    s = text.strip()

    # email: username@hostname.domain (domain in {com, org, edu, gov})
    re_email = re.compile(
        r'^'
        r'[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?'  # username
        r'@'
        r'[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?'  # hostname (can include dots)
        r'\.(com|org|edu|gov)'
        r'$'
    )
    if re_email.fullmatch(s):
        return EMAIL

    # date: YYYY/MM/DD or YY/MM/DD, YYYY-MM-DD or YY-MM-DD
    re_date = re.compile(
        r'^(?P<year>\d{2}|\d{4})(?P<sep>[/-])(?P<month>\d{1,2})(?P=sep)(?P<day>\d{1,2})$'
    )
    m = re_date.fullmatch(s)
    if m:
        y_raw = m.group('year')
        month = int(m.group('month'))
        day = int(m.group('day'))

        if len(y_raw) == 4:
            year = int(y_raw)
        else:
            yy = int(y_raw)
            year = 1900 + yy if yy >= 51 else 2000 + yy  # 51-99 => 1951-1999, 00-50 => 2000-2050

        if 1951 <= year <= 2050:
            try:
                date(year, month, day)  # validates month/day for that month (incl leap years)
                return DATE
            except ValueError:
                pass

    # url: http(s)://address where address has letters/hyphen/dots, starts w/ alnum, includes a dot
    re_url = re.compile(r'^(https?)://(?P<addr>[A-Za-z0-9.-]+)$')
    m = re_url.fullmatch(s)
    if m:
        addr = m.group('addr')
        if '.' in addr and addr[0].isalnum() and addr[-1].isalnum():
            return URL

    # cite:
    # Smith, 2023
    # Smith and Jones, 2023
    # Smith et al., 2023
    last = r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)*"
    name = rf"{last}(?:\s+{last})*"

    re_cite_single = re.compile(rf'^(?P<a>{name}),\s*(?P<y>\d{{4}})$')
    re_cite_two = re.compile(rf'^(?P<a>{name})\s+and\s+(?P<b>{name}),\s*(?P<y>\d{{4}})$')
    re_cite_multi = re.compile(rf'^(?P<a>{name})\s+et\s+al\.,\s*(?P<y>\d{{4}})$')

    for rc in (re_cite_two, re_cite_multi, re_cite_single):
        m = rc.fullmatch(s)
        if m:
            year = int(m.group('y'))
            if 1900 <= year <= 2024:
                return CITE

    # Task 2
    return None


if __name__ == '__main__':
    filepath = 'dat/chronicles_of_narnia.txt'
    d = chronicles_of_narnia(filepath)

    print("num books =", len(d))
    for title, info in d.items():
        print(title, info['year'], len(info['chapters']))

    regex_tests = [
        'anushka.basu@emory.edu',
        '2026/2/16',
        'http://www.emory.edu',
        'Smith, 2024'
    ]
    for test in regex_tests:
        print(regular_expressions(test))