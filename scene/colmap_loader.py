#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import collections
import struct
from pathlib import Path

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_cameras_text(cameras, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")

def write_cameras_text_slam(cameras, path, img_idx):
    """
    This function updates a text file containing camera info. It updates the header with
    the current number of cameras (img_idx+1) and appends the new camera line(s) to the file.
    
    The header is assumed to be the first three lines.
    """
    # Prepare new camera lines to append
    new_camera_lines = []
    for _, cam in cameras.items():
        # Combine camera attributes and parameters into a single line
        to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
        line = " ".join(str(elem) for elem in to_write) + "\n"
        new_camera_lines.append(line)

    # If the file exists, update header and append new data
    if Path(path).exists():
        with open(path, "r") as fid:
            lines = fid.readlines()
        
        # Update the header (first three lines) with the new camera count
        # (Assumes header is exactly three lines)
        header = lines[:3]
        header[2] = "# Number of cameras: {}\n".format(img_idx)
        
        # Get previously saved camera lines (if any)
        camera_lines = lines[3:]
        
        # Append new camera info
        camera_lines.extend(new_camera_lines)
        
        # Write updated header and camera lines back to the file
        with open(path, "w") as fid:
            fid.write("".join(header))
            fid.write("".join(camera_lines))
    else:
        # Create a new file with header and the new camera info
        HEADER = (
            "# Camera list with one line of data per camera:\n"
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            "# Number of cameras: {}\n".format(img_idx)
        )
        with open(path, "w") as fid:
            fid.write(HEADER)
            for line in new_camera_lines:
                fid.write(line)



def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras

def write_cameras_binary_slam(cameras, path_to_model_file, img_idx):
    """
    Updates the binary camera file by appending new camera data and updating the header count.
    
    see: src/colmap/scene/reconstruction.cc
         void Reconstruction::WriteCamerasBinary(const std::string& path)
         void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    # Check if file exists to decide the file mode:
    file_exists = os.path.exists(path_to_model_file)
    mode = "r+b" if file_exists else "wb"
    
    with open(path_to_model_file, mode) as fid:
        if file_exists:
            # File exists: read and update the header count
            fid.seek(0)
            count_bytes = fid.read(8)
            old_count = struct.unpack("Q", count_bytes)[0]
            # Increase the count by the number of new cameras (typically one new camera per call)
            new_count = old_count + len(cameras)
            # Write the updated count at the beginning of the file
            fid.seek(0)
            write_next_bytes(fid, new_count, "Q")
            # Seek to the end to append new camera records
            fid.seek(0, os.SEEK_END)
        else:
            # File doesn't exist: write header with the initial count
            write_next_bytes(fid, len(cameras), "Q")
        
        # Append new camera records
        for _, cam in cameras.items():
            # Get the model id from a predefined mapping
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    
    return cameras



def write_images_text(images, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum(
            (len(img.point3D_ids) for _, img in images.items())
        ) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(images), mean_observations
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [
                img.id,
                *img.qvec,
                *img.tvec,
                img.camera_id,
                img.name,
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")

def write_images_text_slam(images, path, img_idx):
    """
    Updates the images.txt file by recalculating the header with the updated total number
    of images and mean observations per image, then appending the new image records.
    
    File format:
      - 4-line header:
          Line 1: Description
          Line 2: Column names for the image header line
          Line 3: Explanation for the 2D points line
          Line 4: "# Number of images: <total>, mean observations per image: <mean>"
      - Then, for each image, two lines:
          First line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
          Second line: List of 2D points as "X Y POINT3D_ID" (or empty if none)
    """
    path = Path(path)
    
    # Helper: Format the header given total count and mean observations.
    def format_header(total_count, mean_obs):
        return (
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            "# Number of images: {}, mean observations per image: {}\n".format(total_count, mean_obs)
        )
    
    # First, calculate statistics for the new images in 'images'
    new_total_obs = sum(len(img.point3D_ids) for img in images.values())
    new_count = len(images)
    
    if path.exists():
        # Read the existing file to retrieve previous image records
        with open(path, "r") as fid:
            lines = fid.readlines()
        
        # Assume header occupies the first 4 lines
        header_lines = lines[:4]
        data_lines = lines[4:]
        
        # Determine how many images were previously stored.
        # Each image record uses two lines.
        old_count = len(data_lines) // 2
        
        # Compute total observations from the existing records.
        old_total_obs = 0
        for i in range(old_count):
            obs_line = data_lines[2 * i + 1].strip()  # second line for each image record
            if obs_line:  # if there are any points listed
                # Each observation is a triplet: X, Y, POINT3D_ID
                obs_vals = obs_line.split()
                obs_count = len(obs_vals) // 3
                old_total_obs += obs_count
        
        # New totals across all images
        total_count = old_count + new_count
        total_obs = old_total_obs + new_total_obs
        mean_obs = total_obs / total_count if total_count > 0 else 0
        
        # Update header with the new total and mean observation values.
        new_header = format_header(total_count, mean_obs)
        
        # Open the file for writing: update header and keep existing data, then append new image records.
        with open(path, "w") as fid:
            fid.write(new_header)
            # Write back the previous image records
            fid.write("".join(data_lines))
            # Append new images
            for key, img in images.items():
                # Build the image header line. Here we use the image's id attribute if available,
                # otherwise use the provided key (img_idx is typically the new image's index).
                image_id = img_idx # img.id if hasattr(img, "id") else key
                image_header = [
                    image_id,
                    *img.qvec,
                    *img.tvec,
                    img.camera_id,
                    img.name,
                ]
                first_line = " ".join(map(str, image_header)) + "\n"
                fid.write(first_line)
    
                # Build the second line with 2D point observations.
                points_strings = []
                for xy, point3D_id in zip(img.xys, img.point3D_ids):
                    points_strings.append(" ".join(map(str, [*xy, point3D_id])))
                second_line = " ".join(points_strings) + "\n"
                fid.write(second_line)
    else:
        # File does not exist, so create a new one with the header based solely on the new images.
        total_count = new_count
        total_obs = new_total_obs
        mean_obs = total_obs / total_count if total_count > 0 else 0
        new_header = format_header(total_count, mean_obs)
        
        with open(path, "w") as fid:
            fid.write(new_header)
            for key, img in images.items():
                image_id = img_idx # img.id if hasattr(img, "id") else key
                image_header = [
                    image_id,
                    *img.qvec,
                    *img.tvec,
                    img.camera_id,
                    img.name,
                ]
                first_line = " ".join(map(str, image_header)) + "\n"
                fid.write(first_line)
    
                points_strings = []
                for xy, point3D_id in zip(img.xys, img.point3D_ids):
                    points_strings.append(" ".join(map(str, [*xy, point3D_id])))
                second_line = " ".join(points_strings) + "\n"
                fid.write(second_line)
                
def write_images_binary(images, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")
            
def write_images_binary_slam(images, path_to_model_file, img_idx):
    """
    Updates the images.bin file by appending new image records and updating the header count.
    
    see: src/colmap/scene/reconstruction.cc
         void Reconstruction::ReadImagesBinary(const std::string& path)
         void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    file_exists = os.path.exists(path_to_model_file)
    mode = "r+b" if file_exists else "wb"
    
    with open(path_to_model_file, mode) as fid:
        if file_exists:
            # File exists: read and update the header count
            fid.seek(0)
            count_bytes = fid.read(8)
            old_count = struct.unpack("Q", count_bytes)[0]
            new_count = old_count + len(images)
            # Write the updated count back at the beginning
            fid.seek(0)
            write_next_bytes(fid, new_count, "Q")
            # Move to the end of the file for appending new records
            fid.seek(0, os.SEEK_END)
        else:
            # New file: write header with the number of images
            write_next_bytes(fid, len(images), "Q")
        
        # Append new image records
        for _, img in images.items():
            # Write IMAGE_ID (using the provided img_idx)
            write_next_bytes(fid, img_idx, "i")
            # Write quaternion vector (qvec) as 4 doubles
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            # Write translation vector (tvec) as 3 doubles
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            # Write camera_id as an integer
            write_next_bytes(fid, img_idx, "i")
            # Write the image name as a null-terminated string (each char separately)
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            # Write the null terminator
            write_next_bytes(fid, b"\x00", "c")
            # Write the number of 2D points as an unsigned long long ("Q")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            # Write each 2D point record: two doubles for (x, y) and one long long for the 3D point id
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")
    
    return images

def write_points3D_text(points3D, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum(
            (len(pt.image_ids) for _, pt in points3D.items())
        ) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(
            len(points3D), mean_track_length
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")