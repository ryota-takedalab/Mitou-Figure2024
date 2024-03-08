import numpy as np

H36M_KEY = {
    "MidHip": 0,
    "RHip": 1,
    "RKnee": 2,
    "RAnkle": 3,
    "LHip": 4,
    "LKnee": 5,
    "LAnkle": 6,
    "Spine": 7,
    "Thorax": 8,
    "Nose": 9,
    "Head": 10,
    "LShoulder": 11,
    "LElbow": 12,
    "LWrist": 13,
    "RShoulder": 14,
    "RElbow": 15,
    "RWrist": 16,
}

H36M_BONE = np.array(
    [
        [H36M_KEY["Head"], H36M_KEY["Nose"]],
        [H36M_KEY["Nose"], H36M_KEY["Thorax"]],
        [H36M_KEY["Thorax"], H36M_KEY["Spine"]],
        [H36M_KEY["Thorax"], H36M_KEY["RShoulder"]],
        [H36M_KEY["Thorax"], H36M_KEY["LShoulder"]],
        [H36M_KEY["RShoulder"], H36M_KEY["RElbow"]],
        [H36M_KEY["LShoulder"], H36M_KEY["LElbow"]],
        [H36M_KEY["RWrist"], H36M_KEY["RElbow"]],
        [H36M_KEY["LWrist"], H36M_KEY["LElbow"]],
        [H36M_KEY["Spine"], H36M_KEY["MidHip"]],
        [H36M_KEY["RHip"], H36M_KEY["MidHip"]],
        [H36M_KEY["LHip"], H36M_KEY["MidHip"]],
        [H36M_KEY["RHip"], H36M_KEY["RKnee"]],
        [H36M_KEY["RKnee"], H36M_KEY["RAnkle"]],
        [H36M_KEY["LHip"], H36M_KEY["LKnee"]],
        [H36M_KEY["LKnee"], H36M_KEY["LAnkle"]],
    ],
    dtype=np.int64,
)

H36M_ALIGNED_KEY = {
    "MidHip": 0,
    "RHip": 1,
    "RKnee": 2,
    "RAnkle": 3,
    "LHip": 4,
    "LKnee": 5,
    "LAnkle": 6,
    "Thorax": 7,
    "Nose": 8,
    "LShoulder": 9,
    "LElbow": 10,
    "LWrist": 11,
    "RShoulder": 12,
    "RElbow": 13,
    "RWrist": 14,
}

H36M_ALIGNED_BONE = np.array(
    [
        [H36M_ALIGNED_KEY["Nose"], H36M_ALIGNED_KEY["Thorax"]],
        [H36M_ALIGNED_KEY["Thorax"], H36M_ALIGNED_KEY["RShoulder"]],
        [H36M_ALIGNED_KEY["Thorax"], H36M_ALIGNED_KEY["LShoulder"]],
        [H36M_ALIGNED_KEY["RShoulder"], H36M_ALIGNED_KEY["RElbow"]],
        [H36M_ALIGNED_KEY["LShoulder"], H36M_ALIGNED_KEY["LElbow"]],
        [H36M_ALIGNED_KEY["RWrist"], H36M_ALIGNED_KEY["RElbow"]],
        [H36M_ALIGNED_KEY["LWrist"], H36M_ALIGNED_KEY["LElbow"]],
        [H36M_ALIGNED_KEY["Thorax"], H36M_ALIGNED_KEY["MidHip"]],
        [H36M_ALIGNED_KEY["RHip"], H36M_ALIGNED_KEY["MidHip"]],
        [H36M_ALIGNED_KEY["LHip"], H36M_ALIGNED_KEY["MidHip"]],
        [H36M_ALIGNED_KEY["RHip"], H36M_ALIGNED_KEY["RKnee"]],
        [H36M_ALIGNED_KEY["RKnee"], H36M_ALIGNED_KEY["RAnkle"]],
        [H36M_ALIGNED_KEY["LHip"], H36M_ALIGNED_KEY["LKnee"]],
        [H36M_ALIGNED_KEY["LKnee"], H36M_ALIGNED_KEY["LAnkle"]],
    ],
    dtype=np.int64,
)

WHOLE_BONE = np.array(
    [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
     [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
     [8, 10], [1, 2], [0, 1], [0, 2],
     [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18],
     [15, 19], [16, 20], [16, 21], [16, 22], [91, 92],
     [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
     [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
     [102, 103], [91, 104], [104, 105], [105, 106],
     [106, 107], [91, 108], [108, 109], [109, 110],
     [110, 111], [112, 113], [113, 114], [114, 115],
     [115, 116], [112, 117], [117, 118], [118, 119],
     [119, 120], [112, 121], [121, 122], [122, 123],
     [123, 124], [112, 125], [125, 126], [126, 127],
     [127, 128], [112, 129], [129, 130], [130, 131],
     [131, 132]]
)
