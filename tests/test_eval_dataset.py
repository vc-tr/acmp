"""Tests for YOLO-format dataset export."""

from acmp.eval.dataset import write_yolo_dataset
from acmp.eval.metrics import box_iou
from acmp.eval.synthetic import generate_dataset, yolo_to_boxes


def test_write_yolo_dataset_structure(tmp_path):
    train = generate_dataset(n=3, seed=0)
    val = generate_dataset(n=2, seed=100)
    data_yaml = write_yolo_dataset(train, val, tmp_path)

    assert data_yaml.exists()
    assert "names:" in data_yaml.read_text()
    assert len(list((tmp_path / "images" / "train").glob("*.png"))) == 3
    assert len(list((tmp_path / "labels" / "train").glob("*.txt"))) == 3
    assert len(list((tmp_path / "images" / "val").glob("*.png"))) == 2


def test_labels_match_ground_truth(tmp_path):
    train = generate_dataset(n=1, seed=5)
    write_yolo_dataset(train, [], tmp_path)
    page = train[0]

    label_file = next((tmp_path / "labels" / "train").glob("*.txt"))
    lines = label_file.read_text().strip().splitlines()
    assert len(lines) == len(page.boxes)

    parsed = yolo_to_boxes(lines, page.image.width, page.image.height)
    for gt in page.boxes:
        assert max(box_iou(gt, p) for p in parsed) > 0.99
