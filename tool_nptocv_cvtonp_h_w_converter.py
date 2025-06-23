import re
"""
Tool wenn man Rechtecke-ROIs benutzt, opencv benutzt andere schreibweise als roi=frame[numpy schreibweise]
"""
def region_to_numpy(x, y, w, h):
    x1 = x
    x2 = x + w
    y1 = y
    y2 = y + h
    return f"frame[{y1}:{y2}, {x1}:{x2}]"

def numpy_to_region(y1, y2, x1, x2):
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h

def parse_numpy_syntax(s): # regex
    """
    Eingabe Format [y1:y2, x1:x2] oder [ y1 : y2 , x1 : x2 ]
    """
    pattern = r"\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]"
    match = re.match(pattern, s)
    if not match:
        raise ValueError("Fehler Ungültiges Format. Beispiel: [285:575, 0:480]")

    y1, y2, x1, x2 = map(int, match.groups())
    return numpy_to_region(y1, y2, x1, x2)

def main():
    print("Gib entweder ein:")
    print("  - (x, y, w, h)      → für Opencv-Rechteckangabe")
    print("  - (y1, y2, x1, x2)  → für NumPy-Slices-Schreibweise")

    print("1: (x, y, w, h) → NumPy")
    print("2: NumPy → (x, y, w, h)")
    choice = input("Wähle 1 oder 2: ").strip()

    if choice == "1":
        try:
            inp = input("Eingabe: ").strip()
            values = eval(inp)  # vorsichtig bei eval – hier aber zweckmäßig
            if not isinstance(values, tuple) or len(values) != 4:
                print("Fehler Bitte genau vier Werte eingeben, z.B. (0, 285, 480, 290)")
                return
            a, b, c, d = values
            print("NumPy-Syntax:", region_to_numpy(a, b, c, d))
        except Exception as e:
            print("Fehler Fehlerhafte Eingabe:", e)
    elif choice == "2":
        user_input = input("Eingabe: ").strip()
        try:
            x, y, w, h = parse_numpy_syntax(user_input)
            print(f"({x}, {y}, {w}, {h})")
        except ValueError as e:
            print(str(e))


if __name__ == "__main__":
    main()
