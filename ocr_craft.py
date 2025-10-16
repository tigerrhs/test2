# -*- coding: utf-8 -*-
import os
from typing import Tuple
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn.functional as F

import cv2
import numpy as np
from craft.imgproc import loadImage
from craft.detection import get_detector, test_net
from recognition.recognition import get_recognizer
from recognition.dataset import AlignCollate, EditorPostRGBDataset

from common_module import write_log, get_logger
from configs import etc_config, ocr_config
import ocr.line_sort as line_sort
import ocr.ocr_meta as ocr_meta
from to_image import get_image_size

import threading
from file_manager import make_directory

model_lock = threading.Semaphore()
model_lock = threading.Lock()

detector = None
recognizer = None
converter = None
refine_net = None

""" 
** icu 설치 오류에 대한 대처 **
brew install pkg-config icu4c

export PATH="/usr/local/opt/icu4c/bin:/usr/local/opt/icu4c/sbin:$PATH"
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:/usr/local/opt/icu4c/lib/pkgconfig"

pip install --no-binary=:pyicu: pyicu

sudo port install py39-pyicu 

python3.9 -m pip install pyicu==2.9
"""

CSGT = float(ocr_config['CONFIDENCE_SCORE_GREATER_THAN'])
logger = get_logger()


def safe_rect(img, box):
    poly = np.array(box).astype(np.int32).reshape((-1))
    poly = poly.reshape(-1, 2)
    points = poly.reshape((-1, 1, 2))
    rect = cv2.boundingRect(points)
    x, y, w, h = rect

    # 이미지의 크기 확인
    height, width = img.shape[:2]
    
    # x, y 좌표가 음수인 경우 0으로 조정
    x = max(0, x)
    y = max(0, y)
    
    # 너비와 높이가 이미지 경계를 넘어가지 않도록 조정
    w = min(width - x, w)
    h = min(height - y, h)
    
    return (x, y, w, h)


# OCR 이미지 Crop 처리 및 저장
def crop_image(image_file_path, image, bboxes) -> Tuple[list, str]:
    # 이미지 경로를 사용하여 OCR 경로를 생성 (source_ocr 사용 안함)
    ocrCropPath = image_file_path.replace("Image", "OCR").replace("_", "/")
    ocrCropPath = os.path.splitext(ocrCropPath)[0]

    make_directory(ocrCropPath)

    for i, box in enumerate(bboxes):
        x, y, w, h = box

        img_file_name = ocrCropPath + '/' + str(i) + '.png'         # Editor 용 cropped image 저장 경로
        cropped_rect = image[y:y + h, x:x + w]

        if cropped_rect is not None:
            cv2.imwrite(img_file_name, cropped_rect)

    return ocrCropPath

# OCR 모델 로드
def load_models(opt, uid) -> str:
    global detector, refine_net, recognizer, converter

    errCode = ''

    # 탐지 모델, 인식 모델은 한번만 읽어오고 공용으로 사용하기 때문에 동기화
    if detector is not None and recognizer is not None:
        return errCode
    
    write_log("[OCR] load_models", etc_config['LOG_LEVEL_INFO'], uid)

    with model_lock:
        if detector is None:
            detector, refine_net = get_detector(opt.detector, opt.cuda, opt.refine, opt.refiner)
        if recognizer is None:
            recognizer, converter = get_recognizer(opt)

        if detector == None or recognizer == None:
            errCode = 'E904'

    return errCode


def detect_run(opt, image):
    return test_net(detector, image, opt.text_threshold, opt.link_threshold, opt.low_text, opt.cuda, opt.canvas_size, opt.mag_ratio, opt.refine_net)


def classfy_run(opt, pageNo, image_file_path, bboxes, image, scale) -> Tuple[str, int, float, dict]:
    # detect 한 문자들을 crop 하고 파일로 저장

    bboxes = [safe_rect(image, box) for box in bboxes]  # (x, y, w, h) * scale

    if ocr_config['debug_mode'] == 'True' and len(bboxes) > 0:
        try:
            ocrCropPath = crop_image(image_file_path, image, bboxes)    # (y1, y2, x1, x2) * scale
        except Exception as e:
            logger.error(e)
            return ('E908', None)
    else:   # debug mode = False이거나 detect된 글자가 없을 때
        ocrCropPath = ''

    try:
        # FIELD RELM
        relm_data = EditorPostRGBDataset(image, bboxes, opt, scale)  # x, y, w, h
    except Exception as e:
        logger.error(e)
        return ('E910', None)

    # prepare data
    try:
        h, w = get_image_size(image_file_path, scale)
        # opt.canvas_size = image_h  # <-- 실제 크기로 치환
        # print(f"width: {image_w},    height: {image_h}")

        # Editor 에서 정의한 FIELD_RELM 에 맞게 구해진 boxes  <-- 이걸 JSON 에 담아 리턴해야 함!
        edit_boxes = relm_data.edit_boxes
        AlignCollate_data = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    except Exception as e:
        logger.error(e)
        return ('E909', None)

    try:
        data_loader = torch.utils.data.DataLoader(relm_data, batch_size=opt.batch_size,
                                                  shuffle=False,
                                                  num_workers=int(opt.workers),
                                                  collate_fn=AlignCollate_data, pin_memory=True)

        # predict
        recognizer.eval()

        fields = []
        number = 0

        with torch.no_grad():
            for image_tensors, image_path_list in data_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(opt.device)
                # For max length prediction
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(opt.device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(opt.device)

                if 'CTC' in opt.Prediction:
                    preds = recognizer(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = converter.decode(preds_index, preds_size)
                else:
                    preds = recognizer(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    # preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)

                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    try:
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    except:
                        confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])

                    # 정확도가 50% 이하라면 텍스트 아닐 수 있다.
                    if float(f'{confidence_score:0.4f}') > CSGT:
                        # FIELDS에 데이터 기입
                        field_relm = edit_boxes[number]
                        field = ocr_meta.create_field(img_name, pred, field_relm, w, h)
                        fields.append(field)

                    number = number + 1

            # 텍스트 라인 정렬
            fields = line_sort.run(fields)

        pageData = ocr_meta.create_page(pageNo, image_file_path, ocrCropPath, w, h)
        ocr_meta.set_fields(pageData, fields)

        # print(f"PAGE_PATH: {image_file_path}")
        # print(f"PAGE_WIDTH: {w}")
        # print(f"PAGE_HEIGHT: {h}")
    except Exception as e:
        logger.error(e)
        return ('E911', None)
    return ('', pageData)


def load_image(image_path, title=False):
    img = loadImage(image_path)

    if title:
        h = img.shape[0]
        img = img[:int(h * float(ocr_config['TITLE_RATIO'])), :]
    return img