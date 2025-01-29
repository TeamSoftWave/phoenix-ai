import cv2, time

im = cv2.imread("logo.jpg")
new_im = cv2.resize(im, (0, 0), cv2.INTER_LANCZOS4, fx=0.4, fy=0.2)

def print_pixel(r, g, b):
    print('\x1b[48;2;{};{};{}m\x1b[38;2;{};{};{}m▄'.format(r, g, b, r, g, b), end="")

def print_title():
    for row in new_im:
        print()
        for column in row:
            print_pixel(*column[::-1])

    print("\033[0;0m")
