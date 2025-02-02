from PIL import Image


def MandelbrotPoints(iteration_maximum: int,
                     c: complex):
    z = 0
    stability_value = 1
    for i in range(iteration_maximum):
        z = z ** 2 + c
        if abs(z) > 2.0:
            stability_value = i / iteration_maximum
            return stability_value
    return stability_value




for i in range(100):

    image = Image.new(mode="RGB", size=(1920, 1080))

    # center_point = 0 + 0j
    center_point = -0.761577 - 0.08472907899j
    # center_point = -0.10715093446959 - 0.91210639325904j
    # center_point = 0.25100997358377 + 0.000063j

    desired_width = (1 / (2 ** (
            i + 1)))

    scale = desired_width / image.width
    final_width = scale * image.width
    final_height = scale * image.height

    for a in range(
            image.width):
        for b in range(
                image.height):
            pixel = image.getpixel((a, b))
            complex_coordinate = complex(a, -b) * scale + center_point + complex(-final_width, final_height) / 2
            instability_value = 1 - MandelbrotPoints(iteration_maximum=1000, c=complex_coordinate)

            if (instability_value == 0):
                image.putpixel((a, b), (0, 0, 0))
            else:
                image.putpixel((a, b), (int((1 - instability_value) * 255), int((instability_value) * 255),
                                        int(175)))
    print(str(i))
    image.show()
