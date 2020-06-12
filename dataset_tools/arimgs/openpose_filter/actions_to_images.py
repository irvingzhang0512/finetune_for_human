import argparse
import os
from handpose_utils import HandKeyPointCounter
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to-label-file-path",
        type=str,
        default=
        "/hdd02/zhangyiyang/temporal-shift-module/data/AR/label/summary/0426_annotated_to_labels.txt"
    )
    parser.add_argument(
        "--output-file-path",
        type=str,
        default=
        "/hdd02/zhangyiyang/temporal-shift-module/data/AR/label/img_labels/0426_annotated_to_labels.txt"
    )
    parser.add_argument("--save-every-n-steps", type=int, default=1)

    return parser.parse_args()


def _handle_bg_sample(frames_dir, label_id, save_every_n_steps=1):
    if not os.path.exists(frames_dir):
        print(frames_dir, "doesn't exists.")
        return
    img_samples = []
    for idx, img_name in enumerate(os.listdir(frames_dir)):
        if idx % save_every_n_steps != 0:
            continue
        img_path = os.path.join(frames_dir, img_name)
        img_samples.append(img_path + " " + label_id)
    return img_samples


def _handle_hand_sample(frames_dir,
                        counter,
                        label_id,
                        save_every_n_steps=1,
                        bg_label_id="0",
                        thres=15):
    if not os.path.exists(frames_dir):
        print(frames_dir, "doesn't exists.")
        return
    img_samples = []
    for idx, img_name in enumerate(os.listdir(frames_dir)):
        if idx % save_every_n_steps != 0:
            continue
        img_path = os.path.join(frames_dir, img_name)
        cur_cnt = counter(img_path)
        cur_label_id = label_id if cur_cnt >= thres else bg_label_id
        img_samples.append(img_path + " " + cur_label_id)
    return img_samples


def main(args):
    assert os.path.exists(args.to_label_file_path)
    counter = HandKeyPointCounter()

    img_samples = []
    with open(args.to_label_file_path, 'r') as f:
        samples = [sample.strip() for sample in f]
    for sample in tqdm(samples):
        row = sample.split(" ")
        if row[2] == "0":
            img_samples.extend(
                _handle_bg_sample(row[0], row[2], args.save_every_n_steps))
        else:
            img_samples.extend(
                _handle_hand_sample(row[0], counter, row[2],
                                    args.save_every_n_steps))

    with open(args.output_file_path, "w") as f:
        f.writelines("\n".join(img_samples))


if __name__ == '__main__':
    main(_parse_args())
