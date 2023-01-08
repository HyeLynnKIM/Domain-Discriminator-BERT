from HTML_Utils import *
import os
import json
import re
from itertools import product

import tokenization
import numpy as np
import Chuncker

# from HTML_Processor import process_document
import Table_Holder
import Name_Tagging
import Ranking_ids
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
from Table_Holder import detect_num_word, detect_simple_num_word, get_space_of_num, get_space_num_lists

def table_to_2d(table_tag):
    rowspans = []  # track pending rowspans
    rows = table_tag.find_all('tr')

    # first scan, see how many columns we need
    colcount = 0
    for r, row in enumerate(rows):
        cells = row.find_all(['td', 'th'], recursive=False)
        # count columns (including spanned).
        # add active rowspans from preceding rows
        # we *ignore* the colspan value on the last cell, to prevent
        # creating 'phantom' columns with no actual cells, only extended
        # colspans. This is achieved by hardcoding the last cell width as 1.
        # a colspan of 0 means “fill until the end” but can really only apply
        # to the last cell; ignore it elsewhere.
        colcount = max(
            colcount,
            sum(int(c.get('colspan', 1)) or 1 for c in cells[:-1]) + len(cells[-1:]) + len(rowspans))
        # update rowspan bookkeeping; 0 is a span to the bottom.
        rowspans += [int(c.get('rowspan', 1)) or len(rows) - r for c in cells]
        rowspans = [s - 1 for s in rowspans if s > 1]

    # it doesn't matter if there are still rowspan numbers 'active'; no extra
    # rows to show in the table means the larger than 1 rowspan numbers in the
    # last table row are ignored.

    # build an empty matrix for all possible cells
    table = [[None] * colcount for row in rows]

    # fill matrix from row data
    rowspans = {}  # track pending rowspans, column number mapping to count
    for row, row_elem in enumerate(rows):
        span_offset = 0  # how many columns are skipped due to row and colspans
        for col, cell in enumerate(row_elem.find_all(['td', 'th'], recursive=False)):
            # adjust for preceding row and colspans
            col += span_offset
            while rowspans.get(col, 0):
                span_offset += 1
                col += 1

            # fill table data
            rowspan = rowspans[col] = int(cell.get('rowspan', 1)) or len(rows) - row
            colspan = int(cell.get('colspan', 1)) or colcount - col
            # next column is offset by the colspan
            span_offset += colspan - 1
            value = cell.get_text()
            for drow, dcol in product(range(rowspan), range(colspan)):
                try:
                    table[row + drow][col + dcol] = value
                    rowspans[col + dcol] = rowspan
                except IndexError:
                    # rowspan or colspan outside the confines of the table
                    pass

        # update rowspan bookkeeping
        rowspans = {c: s - 1 for c, s in rowspans.items() if s > 1}

    return table

def RepresentsInt(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#name_tagger = Name_Tagging.Name_tagger()
table_holder = Table_Holder.Holder()
chuncker = Chuncker.Chuncker()

max_length = 512

path_dir = './'

sequence_has_ans = np.zeros(shape=[80000, max_length], dtype=np.int32)
segments_has_ans = np.zeros(shape=[80000, max_length], dtype=np.int32)
cols_has_ans = np.zeros(shape=[80000, max_length], dtype=np.int32)
rows_has_ans = np.zeros(shape=[80000, max_length], dtype=np.int32)
mask_has_ans = np.zeros(shape=[80000, max_length], dtype=np.int32)
answer_span = np.zeros(shape=[80000, 2], dtype=np.int32)
answer_texts = np.zeros(shape=[80000], dtype='<U100')

data_num = 0

tokenizer_ = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
tokenizer_.add_tokens('[STA]')
tokenizer_.add_tokens('[END]')

count = 0
false_count = 0
false_count2 = 0
cor = 0
wrong_case = 0

file_name = './VL_tableqa.json'
print(file_name, 'processing....', data_num)

in_path = file_name
data = json.load(open(in_path, 'r', encoding='utf-8'))

for article in data['data']:
    print(count, false_count, false_count2)

    qas = article['paragraphs'][0]['qas']
    question = qas[0]['question']
    answer_text = qas[0]['answers']['text']
    title_data = article['paragraphs'][0]['context'].split('<table><tbody>')[0]
    table_data = '<table>' + str(article['paragraphs'][0]['context'].split('<table>')[1])
    # print(table_data)

    table_data = table_to_2d(BeautifulSoup(table_data, 'html.parser'))
    check = False
    for r, table_line in enumerate(table_data):
        # print(r, table_line)
        for c, td in enumerate(table_line):
            if str(td) == answer_text:
                table_data[r][c] = '[STA] ' + answer_text + ' [END]'
                check = True

    if check is False:
        continue

    query_tokens = []
    query_tokens.append('[CLS]')
    q_tokens = tokenizer_.tokenize(question.lower())
    for tk in q_tokens:
        query_tokens.append(tk)
    query_tokens.append('[SEP]')

    tokens_ = []
    rows_ = []
    cols_ = []

    for i in range(len(table_data)):
        for j in range(len(table_data[i])):
            if table_data[i][j] is not None:
                tokens = tokenizer_.tokenize(table_data[i][j])

                for k, tk in enumerate(tokens):
                    tokens_.append(tk)
                    rows_.append(i)
                    cols_.append(j)

    start_idx = -1
    end_idx = -1

    tokens = []
    rows = []
    cols = []
    segments = []

    for j, tk in enumerate(query_tokens):
        tokens.append(tk)
        rows.append(0)
        cols.append(0)
        segments.append(0)

    for j, tk in enumerate(tokens_):
        if tk == '[STA]':
            start_idx = len(tokens)
        elif tk == '[END]':
            end_idx = len(tokens) - 1
        else:
            tokens.append(tk)
            rows.append(rows_[j] + 1)
            cols.append(cols_[j] + 1)
            segments.append(1)
    ids = tokenizer_.convert_tokens_to_ids(tokens=tokens)

    if end_idx > max_length or start_idx > max_length:
        false_count += 1
        continue

    if start_idx == -1 or end_idx == -1:
        false_count2 += 1
        continue

    length = len(ids)
    if length > max_length:
        length = max_length

    for j in range(length):
        sequence_has_ans[count, j] = ids[j]
        segments_has_ans[count, j] = segments[j]
        cols_has_ans[count, j] = cols[j]
        rows_has_ans[count, j] = rows[j]
        mask_has_ans[count, j] = 1
    answer_span[count, 0] = start_idx
    answer_span[count, 1] = end_idx
    answer_texts[count] = answer_text
    count += 1

np.save('./np/VL_sequence_table_aihub', sequence_has_ans[0:count])
np.save('./np/VL_segments_table_aihub', segments_has_ans[0:count])
np.save('./np/VL_mask_table_aihub', mask_has_ans[0:count])
np.save('./np/VL_rows_table_aihub', rows_has_ans[0:count])
np.save('./np/VL_cols_table_aihub', cols_has_ans[0:count])
np.save('./np/VL_answer_span_table_aihub', answer_span[0:count])
np.save('./np/VL_answer_texts_aihub', answer_texts[0:count])