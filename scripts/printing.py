from utils import *


def display_tif_image(file_path, with_colored_images=True, with_merged_image=False):
    image = tifffile.imread(file_path)
    actin, axon, dendrite = split_tif_image(image)
    print("Image imported: " + file_path)
    # If with_colored_images == False, subplots on one line only.
    plt.subplot(1 + with_colored_images, 3, 1)
    plt.title("Actin")
    plt.imshow(actin)
    plt.subplot(1 + with_colored_images, 3, 2)
    plt.title("Axon")
    plt.imshow(axon)
    plt.subplot(1 + with_colored_images, 3, 3)
    plt.title("Dendrite")
    plt.imshow(dendrite)

    merged, actin, axon, dendrite = get_colored_images(actin, axon, dendrite)
    if with_colored_images:
        plt.subplot(2, 3, 4)
        plt.title("Actin colored")
        plt.imshow(actin)
        plt.subplot(2, 3, 5)
        plt.title("Axon colored")
        plt.imshow(axon)
        plt.subplot(2, 3, 6)
        plt.title("Dendrite colored")
        plt.imshow(dendrite)
    # Show full screen.
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    if with_merged_image:
        plt.figure()
        plt.imshow(merged)
    plt.show()


def display_images_one_by_one(with_colored_images=True, with_merged_image=False):
    for file_path in get_files_path_generator():
        display_tif_image(file_path, with_colored_images, with_merged_image)


# 1041
def print_number_original_files():
    i = 0
    for subdir, dirs, files in os.walk(main_folder_path):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == ".tif":
                i += 1
    print(i)
    return i


def print_images_size():
    for file_path in get_files_path_generator():
        image = tifffile.imread(file_path)
        print(image.shape)
        input()


# (314, 281)
def print_smallest_image_dimension():
    N = print_number_original_files()
    min_rows = np.inf
    min_cols = np.inf
    i = 0
    for file_path in get_files_path_generator():
        print(str(i + 1) + "/" + str(N))
        image = tifffile.imread(file_path)
        # Remove number of channels from shape
        shape = image.shape[1:]
        if min_rows > shape[0]:
            min_rows = shape[0]
        if min_cols > shape[1]:
            min_cols = shape[1]
        i += 1
    print(min_rows)
    print(min_cols)
