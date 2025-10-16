# -*- coding: utf-8 -*-
import datetime
import array
# OCR Meta Json 생성
def create(doc_no:str, doc_path:str, date_time:datetime) -> dict:
    metaData = {}
    # metaData['metaData'] = []
    metaData['DOC_NO'] = doc_no
    metaData['DOC_PATH'] = doc_path
    metaData['REGIST_DT'] = date_time.strftime('%Y-%m-%d %H:%M:%S')
    metaData['UPDT_DT'] = date_time.strftime('%Y-%m-%d %H:%M:%S')
    metaData['PAGES'] = []
    return metaData

def set_result_path(meta_data:dict, resultPath:str):
    meta_data['RESULT_PATH'] = resultPath

def add_page(meta_data:dict, page_data:dict):
    meta_data['PAGES'].append(page_data)

def create_page(page_no:int, image_file_path:str, ocr_crop_path:str, width:float, height:float) -> dict:
    return {"PAGE_NO": page_no,
     "PAGE_PATH": image_file_path,
     "PAGE_CROP_PATH": ocr_crop_path,
     "PAGE_WIDTH": width,
     "PAGE_HEIGHT": height}

def set_fields(page_data:dict, fields:array):
    page_data['FIELDS'] = fields

def skip_page(page_no:int, image_file_path:str, width:float, height:float) -> dict:
    return {"PAGE_NO": page_no,
     "PAGE_PATH": image_file_path,
     "PAGE_WIDTH": width,
     "PAGE_HEIGHT": height,
     "OCR_TYPE": "SKIP"}

def create_field(field_no:str, field_text:str, field_relm:array, width:float, height:float) -> dict:
    return {"FIELD_NO": field_no,
     "FIELD_TEXT": field_text,
     "FIELD_RELM": field_relm,
     "FIELD_RELM_NOM": [field_relm[0] / width,
                        field_relm[1] / height,
                        (field_relm[0] + field_relm[2]) / width,
                        (field_relm[1] + field_relm[3]) / height]}
