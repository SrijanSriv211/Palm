import regex, time, sys, os
ansi_escape = regex.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# kprint -> keep print
# save the text in a text file
def kprint(*text, filename=None, println=True):
    print(*text) if println else None

    if filename is None:
        return

    # save cleaned text to the file
    with open(filename, "a", encoding="utf-8") as f:
        f.write(" ".join(tuple(ansi_escape.sub('', part) for part in text)) + "\n")

def get_bin_path():
    # get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # navigate up the directory tree four times
    parent_dir = current_file_path
    rel_current_path = os.path.relpath(__file__)
    for _ in range(len(rel_current_path.split(os.sep))):
        parent_dir = os.path.dirname(parent_dir)

    # append "cache" directory to the parent directory
    return os.path.join(parent_dir, "bin")

def calc_total_time(seconds):
    # separate the integer part (for hours, minutes, and seconds) from the fractional part (for milliseconds)
    sec_int, millis = divmod(seconds, 1)
    millis = int(millis * 1000) # convert the fractional part to milliseconds

    min, sec = divmod(int(sec_int), 60)
    hour, min = divmod(min, 60)
    hours, minutes, seconds = int(hour), int(min), int(sec)

    t = [
        f"{hours} hour" + ("s" if hours > 1 else "") if hours > 0 else None,
        f"{minutes} minute" + ("s" if minutes > 1 else "") if minutes > 0 else None,
        f"{seconds} second" + ("s" if seconds > 1 else "") if seconds > 0 else None,
        f"{millis} ms" if millis > 0 else None
    ]
    t = list(filter(None, t))

    return ", ".join(t) if t else "0 seconds"
