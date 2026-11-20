"""Data tests on tiny hand-made tensors/files (no download needed)."""

import torch
from PIL import Image

from src import data

# ------------------------------------------------------------ Milestone 0 (should pass now)


def test_image_transform_makes_64x64_from_non_square():
    im = Image.new("RGB", (500, 333), color=(255, 0, 0))
    x = data.image_transform(64)(im)
    assert x.shape == (3, 64, 64) and x.dtype == torch.uint8


def test_range_round_trip():
    x = torch.tensor([0, 128, 255], dtype=torch.uint8)
    y = data.to_model_range(x)
    assert y.min() >= -1 and y.max() <= 1
    assert torch.allclose(data.to_image_range(y) * 255, x.float(), atol=1e-4)


def test_flowers_images_item_shape_and_range():
    imgs = torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8)
    ds = data.FlowersImages(imgs, torch.arange(5))
    x, y = ds[2]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert -1.0 <= x.min() and x.max() <= 1.0 and y == 2


# ------------------------------------------------------------ Milestone 1 (fail until Plan steps 5-6)


def test_load_captions_reads_reed_layout(tmp_path):
    d = tmp_path / "text_c10" / "class_00001"
    d.mkdir(parents=True)
    (d / "image_00001.txt").write_text("a red  flower\nthe petals are red\n")
    caps = data.load_captions(tmp_path / "text_c10")
    assert caps == {"image_00001": ["a red flower", "the petals are red"]}


def test_class_split_is_disjoint_by_class(tmp_path):
    labels = torch.arange(102).repeat_interleave(3)
    split = data.class_split(labels, captions_dir=tmp_path / "text_c10")  # no split files -> seeded fallback
    train_cls, test_cls = set(labels[split["train"]].tolist()), set(labels[split["test"]].tolist())
    assert len(train_cls) == 82 and len(test_cls) == 20 and not (train_cls & test_cls)


def test_class_split_reads_reed_files(tmp_path):
    (tmp_path / "trainvalclasses.txt").write_text("class_00001\nclass_00002\n")
    (tmp_path / "testclasses.txt").write_text("class_00003\n")
    labels = torch.tensor([0, 1, 2, 2])
    split = data.class_split(labels, captions_dir=tmp_path / "text_c10")
    assert split["source"] == "reed"
    assert split["train"].tolist() == [0, 1] and split["test"].tolist() == [2, 3]


def test_check_captions_flags_short_and_missing():
    import pytest
    with pytest.raises(ValueError):
        data.check_captions({"image_00001": ["only one"]}, ["image_00001", "image_00002"], per_image=10)


def test_wrong_caption_comes_from_another_class():
    imgs = torch.zeros(6, 3, 64, 64, dtype=torch.uint8)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    emb = torch.arange(6).float().view(6, 1, 1).expand(6, 10, 512).half()  # row r -> value r
    ds = data.FlowersCaptions(imgs, labels, emb, rows=torch.arange(6))
    for i in range(6):
        _, t, t_wrong, r = ds[i]
        assert int(t[0]) == r
        assert labels[int(t_wrong[0])] != labels[r]
