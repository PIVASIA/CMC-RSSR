from __future__ import print_function

import os
import warnings
import numpy as np
from PIL import Image

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

from util import parse_option
warnings.filterwarnings("ignore")


def main(args):
    # load pre-computed features
    features = np.load(args.features_path)

    # load image set
    with open(args.test_image_list, 'r', encoding="utf-8") as f:
        names = f.readlines()        
    images_to_use = [name.strip() for name in names]

    assert len(images_to_use) == features.shape[0]

    pts_features = []
    pts_labels = []
    for filename, feature in zip(images_to_use, features):
        # read label image
        basename = filename.split(".")[0]
        label_path = os.path.join(args.label_folder, "%s.png" % basename)
        if not os.path.isfile(label_path):
            raise FileNotFoundError("Label's not existed: {0}".format(label_path))
        label = Image.open(label_path).convert('L')
        label = np.array(label)

        # extract points with label
        ys, xs = np.nonzero(label)
        
        pts_label = list(label[ys, xs])
        pts_feat = feature[0][:, ys, xs]

        for i in range(len(pts_label)):
            pts_labels.append(pts_label[i])
            pts_features.append(pts_feat[:, i])

    pts_features = np.array(pts_features)
    pts_labels = np.array(pts_labels)
    print(pts_features.shape, pts_labels.shape)

    unique, counts = np.unique(pts_labels, return_counts=True)
    print(dict(zip(unique, counts)))


    # fit model
    # clf = XGBClassifier()
    # scores = cross_val_score(clf, pts_features, pts_labels, scoring='f1_macro', cv=10)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


if __name__ == '__main__':
    # parse the args
    args = parse_option(False)

    main(args)