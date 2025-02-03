import argparse
from cad_3dto2d.converter import convert_2d_drawing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_file", type=str, required=True)
    args = parser.parse_args()
    output_file = args.step_file.replace(".step", ".png")
    convert_2d_drawing(args.step_file, output_file, add_template=True, x_offset=30)


if __name__ == "__main__":
    main()
