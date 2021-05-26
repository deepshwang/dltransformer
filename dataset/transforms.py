import numpy as np
import utils.data_utils as d_utils
from torchvision import transforms


T_modelnet_train = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter(),
    ]
)

T_modelnet_test = d_utils.PointcloudToTensor()