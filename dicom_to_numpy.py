import os
import argparse
import numpy as np
import pydicom


def save_dataset(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("Created a directory: {}\n".format(args.save_path))

    patients_numbers = sorted([patient for patient in os.listdir(args.data_path)])

    for idx, patient in enumerate(patients_numbers):
        input_paths = os.path.join(args.data_path, patient, "quarter_{}mm".format(args.mm))
        target_paths = os.path.join(args.data_path, patient, "full_{}mm".format(args.mm))

        hu_inputs = hu_transform(make_slices(input_paths))
        hu_targets = hu_transform(make_slices(target_paths))

        for img_idx in range(len(hu_inputs)):

            hu_input = hu_inputs[img_idx]
            hu_target = hu_targets[img_idx]

            if patients_numbers[idx] == args.test_patient:
                data_dir = os.path.join(args.save_path, "test_dataset")
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                    print("Created a directory: {}\n".format(data_dir))
                np.savez("{}/{}/{}_{}".format(args.save_path, "test_dataset", patients_numbers[idx], img_idx), input = hu_input, target = hu_target)

            elif patients_numbers[idx] == args.val_patient:
                data_dir = os.path.join(args.save_path, "validation_dataset")
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                    print("Created a directory: {}\n".format(data_dir))
                np.savez("{}/{}/{}_{}".format(args.save_path, "validation_dataset", patients_numbers[idx], img_idx), input = hu_input, target = hu_target)

            else:
                data_dir = os.path.join(args.save_path, "train_dataset")
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                    print("Created a directory: {}\n".format(data_dir))
                np.savez("{}/{}/{}_{}".format(args.save_path, "train_dataset", patients_numbers[idx], img_idx), input = hu_input, target = hu_target)

        print("Patient {} Done.".format(patient))


def make_slices(paths):
    slices = [pydicom.read_file(os.path.join(paths, path)) for path in os.listdir(paths)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def hu_transform(slices):
    image = np.stack([slice.pixel_array for slice in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dicom to NumPy")

    parser.add_argument('--data_path', type=str, default='/datasets/LDCT_2016/Train/')
    parser.add_argument('--save_path', type=str, default='npz_dataset')
    parser.add_argument('--val_patient', type=str, default='L333')
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--mm', type=int, default=3)
    parser.add_argument('--min_range', type=float, default=-1024.0)
    parser.add_argument('--max_range', type=float, default=3072.0)

    args = parser.parse_args()
    save_dataset(args)
