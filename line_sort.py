# -*- coding: utf-8 -*-
import array

def get_x(info: dict) -> float:
    return info.get("FIELD_RELM")[0]

def get_y(info: dict) -> float:
    return info.get("FIELD_RELM")[1]

def line_resize(line:list):
    if len(line) == 0:
        return
    minY = 10e+10
    maxY = 0
    sumH = 0
    for field in line:
        rect = field['FIELD_RELM']
        minY = min(rect[1], minY)
        maxY = max(rect[1]+rect[3], maxY)
        sumH = sumH+rect[3]
    avgH = sumH/len(line)
    y1 = minY + (maxY-minY)/2.0 - avgH/2.0
    for field in line:
        rect = field['FIELD_RELM']
        field['FIELD_RELM'] = [rect[0], y1, rect[2], avgH]
# 텍스트 라인 정렬
def run(fields:array) -> array:
    fields.sort(key=lambda k: [get_x, get_y])
    line_data = []
    sorted_fileds = []
    crate = 0.5
    lrate = 0.65
    oy = 0
    oh = 0

    for i in fields:
        points = i["FIELD_RELM"]
        y = points[1]
        h = points[3]

        y1 = y + h * crate
        y2 = oy + oh * lrate
        if y2 != 0 and y1 > y2:
            line_data.sort(key=get_x)
            line_resize(line_data)
            for f in line_data:
                sorted_fileds.append(f)
            line_data = []
            line_data.append(i)
        else:
            line_data.append(i)
        oy = y
        oh = h
    for f in line_data:
        line_resize(line_data)
        sorted_fileds.append(f)
    return sorted_fileds
