# -*- coding: utf-8 -*-
import json
import os
import time
from typing import Tuple
import traceback

from reportlab.pdfgen import canvas
from pdfrw import PdfReader

from to_image import get_image_size
from to_pdf import check_ocr_all, create_image_pdf, draw_xobj, put_ocr_fields
from file_manager import ocr_result_PDF2, ocr_result_meta
from ocr import ocr_meta
from onbid.extract_agency_new_cover_page import is_cover_page_ocr, is_cover_page_pdf
from onbid.extract_titles import extract_page_title_ocr, extract_page_title_pdf, filter_title_fields, text_in_content
from onbid.table_utils import check_detail_page

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch

import easydict
from common_module import Status, write_log
from common_cipher_module import response_error_db
from configs import etc_config, ocr_config, pdf_config
from dbquery import ocr_hist_insert
import ocr.ocr_craft as ocr_craft
import threading
from error_message import error_message


# """ OPT 기본 설정값 """
opt = None
# opt_lock = threading.Semaphore()
opt_lock = threading.Lock()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
title_ratio = float(ocr_config['TITLE_RATIO'])

# OCR 실행 후 결과를 JSON 및 PDF (JSON 내장) 저장
def ocr_run(pageNo, image_path, scale, uid, title=False) -> Tuple[dict, str]:
    ocr = ocr_craft

    global opt
    if opt is None:
        error_code = init_opt(uid)
        if error_code:
            insert_error_history(error_code, uid)
            return None, error_code

    errCode = ocr.load_models(opt, uid)
    if errCode:
        return {}, errCode
    
    write_log(f"[OCR] {pageNo}페이지 {'제목' if title else ''}OCR", etc_config['LOG_LEVEL_INFO'], uid)
    
    try:
        image = ocr.load_image(image_path, title)
    except Exception as e:
        write_log(str(e), etc_config['LOG_LEVEL_ERROR'], uid)
        return {}, 'E906'

    # detect
    try:
        bboxes = ocr.detect_run(opt, image)
    except Exception as e:
        write_log(str(e), etc_config['LOG_LEVEL_ERROR'], uid)
        return {}, 'E907'

    # classify
    (errCode, page_data) = ocr.classfy_run(opt, pageNo, image_path, bboxes, image, scale)
    if errCode:
        return {}, errCode
    
    page_data['OCR_TYPE'] = "TITLE" if title else "FULL"

    # JSON 결과값 리턴
    return page_data, errCode

# OPT 설정
def init_opt(uid) -> str:
    write_log("OCR : init_opt", etc_config['LOG_LEVEL_INFO'], uid)
    global opt
    global opt_lock
    with opt_lock:
        if opt is not None:
            return ''
        # opt는 한번만 설정하고 계속 사용해야 함.
        # 모델이 변경되는 경우 서버를 내렸다 다시 올려야 함
        if not opt or opt is None:
            opt = easydict.EasyDict()

            opt.refine_net = None
            opt.canvas_size = 1280

            # cuda / device 재설정
            if torch.cuda.is_available():
                opt.cuda = True
                opt.device = "cuda:0"
            else:
                opt.cuda = False
                opt.device = "cpu"

            opt.workers = 0
            opt.batch_size = 512
            opt.text_threshold = 0.3
            opt.low_text = 0.3
            opt.link_threshold = 0.8

            opt.mag_ratio = 1.0
            opt.refine = False
            opt.sensitive = True
            opt.rgb = 0
            opt.PAD = 0

            opt.FeatureExtraction = 'ResNet'
            opt.Transformation = 'TPS'
            opt.SequenceModeling = 'BiLSTM'
            opt.Prediction = 'Attn'

            opt.num_fiducial = 20
            opt.batch_max_length = 25
            opt.imgH = 32
            opt.imgW = 150
            opt.input_channel = 1
            opt.output_channel = 512
            opt.hidden_size = 256

            opt.character = './OCR_Model/character.txt'
            opt.detector = './OCR_Model/craft_mlt_25k.pth'
            opt.refiner = './OCR_Model/craft_refiner_CTW1500.pth'
            opt.recognizer = './OCR_Model/best_accuracy.pth'

            # open file in read mode
            try:
                with open(opt.character, "r", encoding="utf-8-sig") as characterFile:
                    # read the content of file
                    characterData = characterFile.read()
                    #print(f"characterData: {characterData}")
                    opt.character = characterData
                    characterFile.close()

            except Exception as e:
                opt = None
                print(f"Exception: {e}")
                return 'E912'
    return ''


def insert_error_history(errorCode, uid):
    message = error_message.get(errorCode, '존재하지 않는 에러코드입니다.')
    global opt

    # OCR 이력(MD-05) 삽입         GQ-07
    data = {'DOC_NO': None,  # string
            'META_PTH': None,  # string
            'EXEC_TM': None,  # string Date(HHmmss)
            'SUCCESS_AT': 0,  # <-- 추가
            'ERROR_MESSAGE': message, # <-- 추가
    }  # string
    isHistoryInsertSuccess = ocr_hist_insert(data, uid)

    if not isHistoryInsertSuccess:
        write_log('OCR History Insert Fail', etc_config['LOG_LEVEL_INFO'],uid)
        return response_error_db('E919', 'OCR', 'OCR', uid)  # 'OCR 이력(MD-05) 삽입 실패.'


def skip_ocr(i, image_path, scale, uid):
    h, w = get_image_size(image_path, scale)
    write_log(f'[OCR] 페이지{i} OCR 스킵', etc_config['LOG_LEVEL_INFO'], uid)
    return ocr_meta.skip_page(i, image_path, w, h)


def ocr_main(doc, orgFileName, originalSavePath, orgTimeStr, pdf_status, image_path_list, detectron_results, scale, uid):
    metadata = ocr_meta.create(orgFileName, originalSavePath, orgTimeStr)
    cover_page_idx = None
    summary_page_idx = None
    summary_possible = True
    detail_pages = []
    start_time = time.time()
    all_time=0
    # breakpoint()
    bug_pdf = False
    if doc is not None: # PDF 일 때
        if check_ocr_all(originalSavePath, image_path_list):
            bug_pdf = True
            write_log(f"[OCR] 손상 PDF파일입니다.", etc_config['LOG_LEVEL_WARNING'], uid)
            pdf_status = len(image_path_list) * [Status.RAW]

    for i, image_path in enumerate(image_path_list):
        detail_possible = image_path in detectron_results
        if pdf_status[i] == Status.TEXT:    # 텍스트 페이지
            page_data = skip_ocr(i, image_path, scale, uid)
            page = doc.load_page(i)
            ocr_meta.add_page(metadata, page_data)
            # breakpoint()
            if cover_page_idx is None and is_cover_page_pdf(page):
                cover_page_idx = i
                # return i
                continue

            if summary_possible or detail_possible:
                title_text, title_type = extract_page_title_pdf(page, summary_possible, detail_possible)
                print('1',title_type)
                if title_type:
                    if title_type == "감정평가표":
                        # breakpoint()
                        summary_page_idx = i
                        summary_possible = False
                    elif title_type == "감정평가명세표":
                        detail_pages.append((i, title_text))

                    if not text_in_content(page):                # 제목 부분이 text인데 내용 부분은 이미지
                        s = time.time()
                        page_data, error_code = ocr_run(i, image_path, scale, uid)
                        all_time += time.time()-s
                        if error_code:
                            return ocr_error(error_code, uid)
                        page_data['FIELDS'] = [field for field in page_data['FIELDS'] if field['FIELD_RELM_NOM'][3] > title_ratio]
                        metadata['PAGES'][i] = page_data
                        pdf_status[i] = Status.OCR

                elif detail_possible and len(detectron_results[image_path]) == 1:
                # 제목은 안나왔는데 Detectron돼서 명세표일 수도 있을 때, 명세표는 1페이지에 1개
                    table_rect = [x / scale for x in detectron_results[image_path][0]]
                    if page_data['PAGE_HEIGHT'] * 0.5 < table_rect[3] - table_rect[1] and check_detail_page(page, table_rect):   # 명세표는 큼
                        detail_pages.append((i, ''))

        else:   # 이미지 페이지
            if cover_page_idx is None:  # full ocr
                s = time.time()
                page_data, error_code = ocr_run(i, image_path, scale, uid)
                breakpoint()
                all_time += time.time()-s
                if error_code:
                    return ocr_error(error_code, uid)
                pdf_status[i] = Status.OCR
                if is_cover_page_ocr(page_data['FIELDS']):
                    cover_page_idx = i
                elif summary_possible or detail_possible:
                    title_fields = filter_title_fields(page_data)
                    print('2',summary_possible)
                    title_text, title_type = extract_page_title_ocr(title_fields, summary_possible, detail_possible)
                    print('2',title_type)
                    if title_type:
                        if title_type == "감정평가표":
                            summary_page_idx = i
                            summary_possible = False
                        elif title_type == "감정평가명세표" and detail_possible:
                            detail_pages.append((i, title_text))

            elif summary_possible or detail_possible:
                s = time.time()
                page_data, error_code = ocr_run(i, image_path, scale, uid, title=True)
                all_time += time.time()-s
                if error_code:
                    return ocr_error(error_code, uid)

                # NOTE detail_possible이면 무조건 컬럼명 검사 해야 하는가?
                # print('2',summary_possible)
                title_text, title_type = extract_page_title_ocr(page_data['FIELDS'], summary_possible, detail_possible)
                # print('3',title_type)
                if title_type:
                    if title_type == "감정평가표":
                        summary_page_idx = i
                        summary_possible = False
                    elif title_type == "감정평가명세표" and detail_possible:
                        detail_pages.append((i, title_text))
                    
                    s = time.time()
                    page_data, error_code = ocr_run(i, image_path, scale, uid)
                    all_time += time.time()-s
                    if error_code:
                        return ocr_error(error_code, uid)
                    pdf_status[i] = Status.OCR

            else:
                page_data = skip_ocr(i, image_path, scale, uid)

            ocr_meta.add_page(metadata, page_data)

    if doc: # PDF일 때 닫는다
        doc.close()
    
    # print("summary_page_idx:",summary_page_idx)

    # 표지, 감정평가표 못 찾았을 때
    cover_page_idx = cover_page_idx or 0
    if summary_page_idx is None:
        summary_page_idx = cover_page_idx + 1 if len(pdf_status) > cover_page_idx + 1 else cover_page_idx
        if pdf_status[summary_page_idx] == Status.RAW:
            s = time.time()
            page_data, error_code = ocr_run(summary_page_idx, image_path, scale, uid)
            all_time += time.time()-s
            if error_code:
                return ocr_error(error_code, uid)
            pdf_status[summary_page_idx] = Status.OCR
            metadata['PAGES'][summary_page_idx] = page_data

    required_pages = {cover_page_idx, summary_page_idx, *[p[0] for p in detail_pages]}   # 필요한 모든 페이지

    # 원본 파일이 텍스트 PDF일 때
    if all(status == Status.TEXT for status in pdf_status):
        write_log("[TEXT PDF] 원본 파일이 텍스트 PDF 입니다.", etc_config['LOG_LEVEL_INFO'], uid)
        return originalSavePath, cover_page_idx, summary_page_idx, detail_pages, all_time, None

    result_pdf_path = ocr_result_PDF2(orgTimeStr, orgFileName)  # MF-05

    # PDF 생성
    try:
        if bug_pdf or doc is None:  # 이미지이거나 손상파일
            create_image_pdf(result_pdf_path, metadata, uid)
        else:
            error_code = create_pdf(originalSavePath, result_pdf_path, required_pages, metadata, pdf_status, uid)
            if error_code:
                ocr_error(error_code, uid)
        write_log('[TEXT PDF] ' + result_pdf_path, etc_config['LOG_LEVEL_INFO'], uid)
    except:
        traceback.print_exc()
        write_log('[TEXT PDF] 생성 실패', etc_config['LOG_LEVEL_ERROR'], uid) 
        return ocr_error('E917', uid)
    
    exec_time = time.time() - start_time
    write_log(f'[OCR] {exec_time}초', etc_config['LOG_LEVEL_INFO'], uid)

    # 메타 JSON 생성
    try:
        # s = time.time()
        meta_json_path = create_meta_json(orgTimeStr, orgFileName, result_pdf_path, metadata)
        # all_time += time.time()-s
        write_log('[OCR 메타데이터] ' + meta_json_path, etc_config['LOG_LEVEL_INFO'], uid)
    except Exception as e:
        write_log('[OCR 메타데이터] ' + str(e), etc_config['LOG_LEVEL_ERROR'], uid)
        return ocr_error('E916', uid)
    
    # 이력 삽입
    if ocr_hist(orgFileName, meta_json_path, exec_time, uid):
        write_log('[OCR] 이력 삽입', etc_config['LOG_LEVEL_INFO'], uid)
    else:
        write_log('[OCR] 이력 삽입 실패', etc_config['LOG_LEVEL_ERROR'], uid)
        return ocr_error('E919', uid)

    return result_pdf_path, cover_page_idx, summary_page_idx, detail_pages, all_time, None

def ocr_error(error_code, uid):
    insert_error_history(error_code, uid)
    return None, None, None, None, error_code


def create_meta_json(orgTimeStr, orgFileName, result_pdf_path, metadata):
    meta_json_path = ocr_result_meta(orgTimeStr, orgFileName)
    ocr_meta.set_result_path(metadata, result_pdf_path)
    with open(meta_json_path, 'w', encoding='UTF-8-sig') as outfile:
        json.dump(metadata, outfile, indent=2, ensure_ascii=False)
    return meta_json_path


def ocr_hist(file_name, meta_json_path, exec_time, uid):
    data = {
        'DOC_NO': file_name,  # string
        'META_PTH': meta_json_path,  # string
        'EXEC_TM': exec_time,
        'SUCCESS_AT': 1,
        'ERROR_MESSAGE': ''
    }

    return ocr_hist_insert(data, uid)


def ocr_rotate_page(ocr_pages: set, page_rotation: dict, image_path_list: list, pdf_status: list, metadata: list, uid):
    '''텍스트 페이지에 회전 정보 있는지 확인
    회전 정보 있으면 xobj가 아닌 이미지로 PDF를 생성하기 때문에 OCR 필요'''
    for i in ocr_pages:
        if i in page_rotation:
            page_data, error_code = ocr_run(i, image_path_list[i], pdf_config['PDF_TO_IMAGE_SCALE'], uid)
            if error_code:
                return error_code
            metadata[i] = page_data
            pdf_status[i] = Status.OCR
            write_log(f"[OCR] {i}페이지 OCR (/Rotation: {page_rotation[i]})", etc_config['LOG_LEVEL_INFO'], uid)
    return None


def create_pdf(input_file_path, output_file_path, ocr_pages: set, ocr_meta: dict, pdf_status:list, uid: str):
    """PDF 생성하다가 오류 나면 OCR해야 하기 때문에 to_pdf.py에서 ocr.py로 가져옴"""
    reader = PdfReader(input_file_path, decompress=False)
    c = canvas.Canvas(output_file_path)
    pages_num = len(ocr_meta['PAGES'])
    for i in range(pages_num):
        page_meta = ocr_meta['PAGES'][i]
        if not draw_xobj(c, reader.pages[i], page_meta, uid) and i in ocr_pages and pdf_status[i] == Status.TEXT:
            # 텍스트 페이지였는데 텍스트 정보 없어졌기 때문에 다시 OCR
            page_meta, error_code = ocr_run(i, page_meta['PAGE_PATH'], float(pdf_config['PDF_TO_IMAGE_SCALE']), uid)
            if error_code:
                return error_code
            ocr_meta['PAGES'][i] = page_meta
            pdf_status[i] == Status.OCR

        if i in ocr_pages:
            put_ocr_fields(page_meta, c)
            write_log(f"[draw OCR] {i}페이지", etc_config['LOG_LEVEL_INFO'], uid)
        c.showPage()
    c.save()