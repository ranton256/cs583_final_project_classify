#!/usr/bin/python
import os
import random
import shutil

"""
human = Chihuahua cls = n02085620
human = Yorkshire_terrier cls = n02094433
human = golden_retriever cls = n02099601
human = Labrador_retriever cls = n02099712
human = German_shepherd cls = n02106662
human = standard_poodle cls = n02113799

human = tabby cls = n02123045
human = Persian_cat cls = n02123394
human = Egyptian_cat cls = n02124075
human = cougar cls = n02125311
human = lion cls = n02129165
"""

canine_classes = ["n02085620", "n02094433", "n02099601", "n02099712", "n02106662", "n02113799"]
feline_classes = ["n02123045", "n02123394", "n02124075", "n02125311", "n02129165"]

top_dst_dir="felines_and_canines"


def label_for_class(cls):
    if cls in canine_classes:
        dst_label = 'canine'
    elif cls in feline_classes:
        dst_label = 'feline'
    else:
        dst_label = None
    return dst_label


def get_human_labels():
    with open("keep_labels.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            cls, human = line.split(',')
            human = human.strip()
            cls = cls.strip()
            print("human = {} cls = {}".format(human, cls))


def setup_training_data():
    src_datadirs=["train"]

    if not os.path.exists(top_dst_dir):
        os.makedirs(top_dst_dir)

    for src_dir in src_datadirs:
        bname = os.path.basename(src_dir)
        print("bname",bname)
        if not os.path.exists(top_dst_dir):
            os.makedirs(top_dst_dir)
        dst_dir = os.path.join(top_dst_dir, bname)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        print("dst_dir", dst_dir)
        for src_fname in os.listdir(src_dir):
            dst_label = label_for_class(src_fname)
            if not dst_label:
                continue

            dst_label_dir = dst_path = os.path.join(dst_dir,dst_label)
            if not os.path.exists(dst_label_dir):
                os.makedirs(dst_label_dir)

            image_src_dir = os.path.join(src_dir, src_fname, "images")
            for image_fname in os.listdir(image_src_dir):
                src_path = os.path.join(image_src_dir, image_fname)
                dst_path = os.path.join(dst_label_dir, image_fname)
                print("copy {} to {}".format(src_path, dst_path))

                try:
                    shutil.copy(src_path, dst_path)
                except IOError as e:
                    print("Error copying file from {} to {}: {}".format(src_path, dst_path, e))



def read_validation_labels(val_labels_path):
    labels = {}
    with open(val_labels_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            label = label_for_class(parts[1])
            if label:
                labels[parts[0]] = label
    return labels


def setup_validation_data():
    validation_dir = "val"
    test_data_dir = "test"
    image_src_dir = os.path.join(validation_dir, "images")

    labels = read_validation_labels(os.path.join(validation_dir, "val_annotations.txt"))
    print("# validation images=", len(labels))

    for image_fname in os.listdir(image_src_dir):
        if not image_fname in labels:
            continue # not part of our subset

        dst_label = labels[image_fname]
        dst_label_dir = os.path.join(top_dst_dir, validation_dir, dst_label)
        if not os.path.exists(dst_label_dir):
            os.makedirs(dst_label_dir)
        src_path = os.path.join(image_src_dir, image_fname)
        dst_path = os.path.join(dst_label_dir, image_fname)
        print("copy {} to {}".format(src_path, dst_path))

        try:
            shutil.copy(src_path, dst_path)
        except IOError as e:
            print("Error copying file from {} to {}: {}".format(src_path, dst_path, e))


def setup_test_data():
    train_data_dir = "train"
    test_data_dir = "test"
    test_percentage = 10

    # We are going to have to use part of training data for testing
    # because we don't have labeled data for original test folder of dataset.
    # so MOVE 10% of training data out to test folder

    class_names = ["feline","canine"]
    for cls in class_names:
        image_src_dir = os.path.join(top_dst_dir, train_data_dir, cls)
        print("image_src_dir",image_src_dir)
        image_dst_dir = os.path.join(top_dst_dir, test_data_dir, cls)
        print("image_dst_dir", image_dst_dir)
        if not os.path.exists(image_dst_dir):
            os.makedirs(image_dst_dir)
        # create list of files in this class and randomly choose % of them
        cls_files = [fn for fn in os.listdir(image_src_dir)]
        random.shuffle(cls_files)
        test_cnt = len(cls_files) * test_percentage / 100
        test_cnt = int(test_cnt)
        print("choosing {} images for test in class {}".format(test_cnt,cls))
        test_cls_files = cls_files[:test_cnt]
        for image_fname in test_cls_files:
            src_path = os.path.join(image_src_dir, image_fname)
            dst_path = os.path.join(image_dst_dir, image_fname)
            print("move {} to {}".format(src_path, dst_path))
            try:
                shutil.move(src_path, dst_path)
            except IOError as e:
                print("Error moving file from {} to {}: {}".format(src_path, dst_path, e))



def main():
    setup_training_data()
    setup_validation_data()
    setup_test_data()


if __name__ == "__main__":
    main()