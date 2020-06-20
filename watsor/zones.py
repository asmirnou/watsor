import cv2
import sys
from argparse import ArgumentParser
from watsor.filter.mask import get_alpha_channel, find_contours


def parse_args():
    parser = ArgumentParser(description='Indexes zones in mask image depending on '
                                        'the proximity of their center to the origin')
    parser.add_argument('-m', '--mask',
                        dest='mask', metavar='MASK', required=True,
                        help="mask image filename")

    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    return parser.parse_args()


def draw_contours(image, contours, color=(0, 0, 255), thickness=1, font_scale=1):
    cv2.drawContours(image, contours, -1, color, thickness)

    num = 0
    for c in contours:
        num += 1
        display_str = str(num)
        moments = cv2.moments(c)
        center_x = moments['m10'] / moments['m00']
        center_y = moments['m01'] / moments['m00']

        font = cv2.FONT_HERSHEY_DUPLEX
        size = cv2.getTextSize(display_str, font, font_scale, thickness)
        text_width = size[0][0]
        text_height = size[0][1]

        cv2.putText(image, str(num),
                    (int(center_x - text_width / 2), int(center_y + text_height / 2)),
                    font, font_scale, color, thickness, cv2.LINE_AA)


def main():
    try:
        args = parse_args()
        alpha_channel, mask_image = get_alpha_channel(args.mask)
        contours = find_contours(alpha_channel)
        draw_contours(mask_image, contours)

        cv2.imshow(args.mask, mask_image)
        while cv2.waitKey(100) < 0 < cv2.getWindowProperty(args.mask, cv2.WND_PROP_VISIBLE):
            pass
        cv2.destroyAllWindows()
    except AssertionError as e:
        print(e, file=sys.stderr)
    except cv2.error as e:
        print("Error: {}\n{}".format(e, "Make sure 'opencv-python' module is installed"),
              file=sys.stderr)


if __name__ == '__main__':
    main()
