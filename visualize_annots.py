import numpy as np
import cv2
from pycocotools.coco import COCO

"""
1. Open Annotation
2. Read Ground-truth annotation
3. Visualize
"""


class BboxVisualizer:
  def __init__(self, category: object):
    self.category = category
    colors = [(color_list[_]).astype(np.uint8) \
              for _ in range(len(color_list))]
    self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)

  def add_coco_bbox(self, img, bbox, cat, conf=1, show_txt=True):
    img = np.array(img)
    bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    c = self.colors[cat][0][0].tolist()
    name = self.category[cat]['name']
    txt = '{}{:.1f}'.format(name, conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(
      img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    if show_txt:
      cv2.rectangle(img,
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
      cv2.putText(img, txt, (bbox[0], bbox[1] - 2),
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img


color_list = np.array(
  [
    1.000, 1.000, 1.000,
    0.850, 0.325, 0.098,
    0.929, 0.694, 0.125,
    0.494, 0.184, 0.556,
    0.466, 0.674, 0.188,
    0.301, 0.745, 0.933,
    0.635, 0.078, 0.184,
    0.300, 0.300, 0.300,
    0.600, 0.600, 0.600,
    1.000, 0.000, 0.000,
    1.000, 0.500, 0.000,
    0.749, 0.749, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 1.000,
    0.667, 0.000, 1.000,
    0.333, 0.333, 0.000,
    0.333, 0.667, 0.000,
    0.333, 1.000, 0.000,
    0.667, 0.333, 0.000,
    0.667, 0.667, 0.000,
    0.667, 1.000, 0.000,
    1.000, 0.333, 0.000,
    1.000, 0.667, 0.000,
    1.000, 1.000, 0.000,
    0.000, 0.333, 0.500,
    0.000, 0.667, 0.500,
    0.000, 1.000, 0.500,
    0.333, 0.000, 0.500,
    0.333, 0.333, 0.500,
    0.333, 0.667, 0.500,
    0.333, 1.000, 0.500,
    0.667, 0.000, 0.500,
    0.667, 0.333, 0.500,
    0.667, 0.667, 0.500,
    0.667, 1.000, 0.500,
    1.000, 0.000, 0.500,
    1.000, 0.333, 0.500,
    1.000, 0.667, 0.500,
    1.000, 1.000, 0.500,
    0.000, 0.333, 1.000,
    0.000, 0.667, 1.000,
    0.000, 1.000, 1.000,
    0.333, 0.000, 1.000,
    0.333, 0.333, 1.000,
    0.333, 0.667, 1.000,
    0.333, 1.000, 1.000,
    0.667, 0.000, 1.000,
    0.667, 0.333, 1.000,
    0.667, 0.667, 1.000,
    0.667, 1.000, 1.000,
    1.000, 0.000, 1.000,
    1.000, 0.333, 1.000,
    1.000, 0.667, 1.000,
    0.167, 0.000, 0.000,
    0.333, 0.000, 0.000,
    0.500, 0.000, 0.000,
    0.667, 0.000, 0.000,
    0.833, 0.000, 0.000,
    1.000, 0.000, 0.000,
    0.000, 0.167, 0.000,
    0.000, 0.333, 0.000,
    0.000, 0.500, 0.000,
    0.000, 0.667, 0.000,
    0.000, 0.833, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 0.167,
    0.000, 0.000, 0.333,
    0.000, 0.000, 0.500,
    0.000, 0.000, 0.667,
    0.000, 0.000, 0.833,
    0.000, 0.000, 1.000,
    0.000, 0.000, 0.000,
    0.143, 0.143, 0.143,
    0.286, 0.286, 0.286,
    0.429, 0.429, 0.429,
    0.571, 0.571, 0.571,
    0.714, 0.714, 0.714,
    0.857, 0.857, 0.857,
    0.000, 0.447, 0.741,
    0.50, 0.5, 0
  ]
).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255


def loadAnnot(annot_path, img_idx):
  coco = COCO(annot_path)
  ImgId = coco.getImgIds()[img_idx]
  img = coco.loadImgs(ImgId)
  print(img)
  return img


def visualizeRotationResult(annot_path, annot_file, image_path, image_idx, ori_vis=False):
  # first coco image demo showing
  coco = COCO(annot_path + annot_file)
  srcImgId = coco.getImgIds()[image_idx]
  srcImage = coco.loadImgs(srcImgId)[0]

  # open source image
  print(srcImage['file_name'])
  src_image = cv2.imread(image_path + srcImage['file_name'])
  h, w, c = src_image.shape

  if ori_vis:
    cv2.imshow("src_image", src_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

  annotIds = coco.getAnnIds(srcImgId)
  annotations = coco.loadAnns(annotIds)

  bboxVis = BboxVisualizer(coco.loadCats(coco.getCatIds()))
  annot_src_image = np.array(src_image)
  for annot in annotations:
    xmin, ymin, width, height = annot['bbox']
    annot_src_image = bboxVis.add_coco_bbox(annot_src_image,
                                            (xmin, ymin, xmin + width, ymin + height),
                                            annot['category_id'])
  # show source image with annotation
  cv2.imshow("annotated src image", annot_src_image)
  cv2.imwrite("example-output/vis-example.jpg", annot_src_image)
  cv2.waitKey()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  annot_path = "./data/indoor360/data_list/"
  image_path = "./data/indoor360/images_960/"
  annot_file = "train.json"
  image_idx = 1

  print("Start Visualization Demo...")
  visualizeRotationResult(annot_path, annot_file, image_path, image_idx, ori_vis=False)
  print("Finish Visualization Demo!")
