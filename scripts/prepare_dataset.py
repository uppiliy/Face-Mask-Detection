import os, shutil, glob, xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# ==== Update these paths ====
DATASET_DIR = './dataset'
IMG_DIR     = os.path.join(DATASET_DIR, 'images')
XML_DIR     = os.path.join(DATASET_DIR, 'annotations')
BASE        = './yolodata'

# Class label mapping
CLASS_MAP = {"with_mask":0, "without_mask":1, "mask_weared_incorrect":2}

# Create YOLO folder structure
for subset in ['train', 'val', 'test']:
    for kind in ['images', 'labels']:
        os.makedirs(f"{BASE}/{kind}/{subset}", exist_ok=True)

# Split XMLs
xml_files = glob.glob(f"{XML_DIR}/*.xml")
train_xml, temp_xml = train_test_split(xml_files, test_size=0.20, random_state=42)
val_xml, test_xml   = train_test_split(temp_xml, test_size=0.50, random_state=42)

# Convert VOC to YOLO
def convert(xml_list, subset):
    for xml_file in xml_list:
        root = ET.parse(xml_file).getroot()
        fname = root.find('filename').text
        w = int(root.find('size/width').text)
        h = int(root.find('size/height').text)

        src = os.path.join(IMG_DIR, fname)
        dst = os.path.join(BASE, 'images', subset, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            continue

        label_file = os.path.join(BASE, 'labels', subset, fname.rsplit('.', 1)[0] + '.txt')
        with open(label_file, 'w') as f:
            for obj in root.findall('object'):
                cls = CLASS_MAP.get(obj.find('name').text)
                if cls is None: continue
                bnd = obj.find('bndbox')
                ymin, xmin, ymax, xmax = [float(bnd.find(t).text) for t in ('ymin','xmin','ymax','xmax')]
                x_c = ((xmin + xmax) / 2) / w
                y_c = ((ymin + ymax) / 2) / h
                bw  = (xmax - xmin) / w
                bh  = (ymax - ymin) / h
                f.write(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

# Run conversion
convert(train_xml, 'train')
convert(val_xml, 'val')
convert(test_xml, 'test')

print("Dataset converted to YOLO format.")
