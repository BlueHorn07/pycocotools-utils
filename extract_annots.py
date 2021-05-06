import json
from pycocotools.coco import COCO

"""
1. Read annotation file using pycocotools
2. Extract N number of annots from original annotation file. (ex: N=10, 100, 1000)
3. Save new extracted one.
"""


def extractAnnot(annot_path, num, save_path):
  coco = COCO(annot_path)
  new_annot_json = {
    "desc": "extract {} number of annots from {}".format(num, annot_path),
    "images": [],
    "annotations": [],
    "categories": coco.loadCats(coco.getCatIds())
  }

  ImgIds = coco.getImgIds()
  ImgIds = ImgIds[0:num]
  new_annot_json['images'] = coco.loadImgs(ImgIds)

  annotIds = coco.getAnnIds(ImgIds)
  new_annot_json['annotations'] = coco.loadAnns(annotIds)

  with open(save_path, "w") as json_file:
    json.dump(new_annot_json, json_file)

  print("done!")


if __name__ == '__main__':
  annot_path = './data/indoor360/data_list/test.json'
  extract_num = 100
  save_path = './data/indoor360/data_list/test_{}.json'.format(extract_num)

  extractAnnot(annot_path, extract_num, save_path)
