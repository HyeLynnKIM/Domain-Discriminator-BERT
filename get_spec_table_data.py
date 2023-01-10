import json
import numpy as np
from transformers import AutoTokenizer
import Table_Holder
import tokenization
import collections

def read_squad_example(orig_answer_text, answer_offset, paragraph_text):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    start_position = None
    end_position = None

    answer_length = len(orig_answer_text)
    start_position = char_to_word_offset[answer_offset]
    end_position = char_to_word_offset[answer_offset + answer_length - 1]

    # Only add answers where the text can be exactly recovered from the
    # document. If this CAN'T happen it's likely due to weird Unicode
    # stuff so we will just skip the example.
    #
    # Note that this means for training mode, every example is NOT
    # guaranteed to be preserved.
    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
    cleaned_answer_text = " ".join(
        tokenization.whitespace_tokenize(orig_answer_text))
    if actual_text.find(cleaned_answer_text) == -1:
        print("Could not find answer: '%s' vs. '%s'",
              actual_text, cleaned_answer_text)
        return -1, -1, -1

    return doc_tokens, start_position, end_position

def convert_example_to_tokens(question_text,
                              start_position, end_position,
                              doc_tokens, orig_answer_text, doc_stride=128):
    max_seq_length = 512

    query_tokens = tokenizer.tokenize(question_text)

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None

    if True:
        tok_start_position = orig_to_tok_index[start_position]
        if end_position < len(doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    arr_input_ids = []
    arr_segment_ids = []
    arr_start_position = []
    arr_end_position = []

    doc_texts = []
    doc_tokens = []

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None
        if True:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
                # print('answer:', answer_text, tokens[start_position: end_position + 1])
                start_position, end_position = _improve_answer_span(tokens, start_position, end_position, tokenizer,
                                                                    answer_text)
                # print('answer:', answer_text, tokens[start_position: end_position + 1])
                # print('-------------------')
        # print('len:', len(input_ids))

        arr_input_ids.append(input_ids)
        arr_segment_ids.append(segment_ids)
        arr_start_position.append(start_position)
        arr_end_position.append(end_position)

        doc_text = ''
        opened = False
        for doc_token in tokens:
            if opened is True:
                doc_text += doc_token + ' '

            if doc_token == '[SEP]':
                opened = True
        doc_text = doc_text.replace(' ##', '')
        doc_texts.append(doc_text)
        doc_tokens.append(tokens)

    return arr_input_ids, arr_segment_ids, arr_start_position, arr_end_position

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

table_holder = Table_Holder.Holder()

vocab = tokenization.load_vocab(vocab_file='vocab_bigbird.txt')
tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

tokenizer_ = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
tokenizer_.add_tokens('[STA]')
tokenizer_.add_tokens('[END]')

file = './TL_tableqa.json'
# df = pd.read_excel(file)

count = 0
false_count = 0
file_error = 0
no_answer_count = 0

# questions = df['질문']
# table_files = df['대상_테이블_html파일명']
# answer_texts = df['답변']
# answer_positions = df['답변 위치']

max_length = 512
max_data_num = 132000

sequence_has_ans = np.zeros(shape=[max_data_num, 4, max_length], dtype=np.int32)
segments_has_ans = np.zeros(shape=[max_data_num, max_length], dtype=np.int32)
positions_has_ans = np.zeros(shape=[max_data_num, max_length], dtype=np.int32)
ranks_has_ans = np.zeros(shape=[max_data_num, max_length], dtype=np.int32)
names_has_ans = np.zeros(shape=[max_data_num, max_length], dtype=np.int32)
cols_has_ans = np.zeros(shape=[max_data_num, max_length], dtype=np.int32)
rows_has_ans = np.zeros(shape=[max_data_num, max_length], dtype=np.int32)
mask_has_ans = np.zeros(shape=[max_data_num, max_length], dtype=np.int32)
numeric_space = np.zeros(shape=[max_data_num, 10, max_length], dtype=np.int32)
numeric_mask = np.zeros(shape=[max_data_num, 10, max_length], dtype=np.int32)
answer_span = np.zeros(shape=[max_data_num, 2], dtype=np.int32)
answer_text_array = np.zeros(shape=[max_data_num], dtype='<U200')
question_text_array = np.zeros(shape=[max_data_num], dtype='<U200')

############################### For JSON ###################################
data = json.load(open(file, 'r', encoding='utf-8'))

for article in data['data']:
    for para in article['paragraphs']:
        for qa in para['qas']:
            query_text = "".join(qa['question']).replace('  ', ' ').replace('\n', '').strip()

            # 1. 행정, 뉴스
            answer_start = qa['answers']['answer_start']
            answer_text = qa['answers']['text']

            # ## 2. 도서
            # answer_start = qa['answers'][0]['answer_start']
            # answer_text = qa['answers'][0]['text']

            doc = "".join(para['context'])

            start_position = 0
            end_position = 0

            try:
                tokens, start_position, end_position = read_squad_example(orig_answer_text=answer_text,
                                                                          paragraph_text=doc,
                                                                          answer_offset=answer_start)
            except:
                print('!')
                continue

            try:
                input_ids_arrays, input_segments_arrays, start_positions, end_positions = convert_example_to_tokens(
                    query_text,
                    start_position,
                    end_position,
                    tokens,
                    answer_text
                )
            except:
                not_found += 1
                continue

            for s in range(len(input_ids_arrays)):
                input_ids_arr = input_ids_arrays[s]
                input_segments_arr = input_segments_arrays[s]
                start_position = start_positions[s]
                end_position = end_positions[s]

                if end_position >= 512:
                    continue

                input_ids[count] = input_ids_arr
                input_segments[count] = input_segments_arr
                answer_span[count, 0] = start_position
                answer_span[count, 1] = end_position
                count += 1

                if count % 100 == 0:
                    print(count)

print(len(questions))

#for i in range(len(questions) - 300, len(questions)):
for i in range(len(questions)):
    if table_files[i].find('.html') == -1:
        table_files[i] = table_files[i] + '.html'
    try:
        file_path = '/data/KorQuAD2_table_roberta/tables/table_data (1)/'
        table_file = open(file_path + table_files[i], 'r',
                          encoding='utf-8')
    except:
        file_path = '/data/KorQuAD2_table_roberta/tables/'
        print(file_path + table_files[i])
        file_error += 1
        continue

    space_character = str(chr(160))
    my_sapce = str(chr(32))

    table_text = table_file.read()
    table_text = table_text.replace('<th', '<td')
    table_text = table_text.replace('</th', '</td')
    table_text = table_text.replace(space_character, my_sapce)

    table_file.close()

    table_holder.get_table_text(table_text=table_text)

    table_data = table_holder.table_data

    answer_row = answer_positions[i].replace('(', '').replace(')', '').split(',')[0]
    answer_row = int(answer_row)# - 1

    answer_col = answer_positions[i].replace('(', '').replace(')', '').split(',')[1]
    answer_col = int(answer_col)# - 1

    answer_text = str(answer_texts[i])
    answer_text = answer_text.replace('\n', ' ')
    answer_text = answer_text.replace('\t', ' ')
    answer_text = answer_text.replace('  ', ' ')
    answer_text = answer_text.replace('  ', ' ')
    answer_text = answer_text.replace(space_character, my_sapce)

    try:
        cell_text = str(table_data[answer_row][answer_col])
    except:
        try:
            temp = answer_row
            answer_row = answer_col
            answer_col = temp

            cell_text = str(table_data[answer_row][answer_col])
        except:
            continue

    try:
        answer_start = cell_text.find(answer_text)

        if answer_start == -1:
            cell_text = '[STA] ' + cell_text + ' [END]'
        else:
            if cell_text[answer_start + len(answer_text)] == ' ':
                cell_text = '[STA] ' + cell_text + ' [END] '
            else:
                cell_text = '[STA] ' + cell_text + ' [END]'
    except:
        cell_text = '[STA] ' + cell_text + ' [END]'
    table_data[answer_row][answer_col] = cell_text

    try:
        if table_data[answer_row][answer_col] is not None:
            table_data[answer_row][answer_col] = str(table_data[answer_row][answer_col])
            table_data[answer_row][answer_col] = table_data[answer_row][answer_col].replace('\n', ' ')
            table_data[answer_row][answer_col] = table_data[answer_row][answer_col].replace('\t', ' ')
            table_data[answer_row][answer_col] = table_data[answer_row][answer_col].replace('  ', ' ')
            table_data[answer_row][answer_col] = table_data[answer_row][answer_col].replace('  ', ' ')
            table_data[answer_row][answer_col] = table_data[answer_row][answer_col].replace(space_character, my_sapce)

            if str(table_data[answer_row][answer_col]).find(str(answer_text)) == -1:
                no_answer_count += 1
                continue
    except:
        if len(table_data) == 0:
            false_count += 1
            continue

        #print(table_text)
        print('---------------------')
        print(table_files[i])
        print(answer_row, answer_col)
        print(len(table_data), len(table_data[0]))
        false_count += 1
        continue

    question = str(questions[i])

    #data create
    query_tokens = []
    query_tokens.append('[CLS]')
    q_tokens = tokenizer_.tokenize(question.lower())
    for tk in q_tokens:
        query_tokens.append(tk)
    query_tokens.append('[SEP]')

    tokens_ = []
    rows_ = []
    cols_ = []
    spaces_ = []

    # name_tags_ = []
    # ranks_ = []
    positions_ = []

    for x in range(len(table_data)):
        for y in range(len(table_data[x])):
            if table_data[x][y] is not None:
                tokens = tokenizer_.tokenize(table_data[x][y])
                # name_tag = name_tagger.get_name_tag(table_data[i][j])

                for k, tk in enumerate(tokens):
                    tokens_.append(tk)
                    rows_.append(x + 1)
                    cols_.append(y)
                    positions_.append(k)

                    spaces_.append(-1)

                    if k >= 40:
                        break

                if len(tokens) > 40 and str(table_data[x][y]).find('[END]') != -1:
                    tokens_.append('[END]')
                    rows_.append(x + 1)
                    cols_.append(y)
                    positions_.append(0)
                    spaces_.append(-1)

    tokens = []
    rows = []
    cols = []
    # ranks = []
    segments = []
    # name_tags = []
    positions = []
    spaces = []

    for j, tk in enumerate(query_tokens):
        tokens.append(tk)
        rows.append(0)
        cols.append(0)
        # ranks.append(0)
        segments.append(0)
        # name_tags.append(0)
        positions.append(j)
        spaces.append(-1)

    start_idx = -1
    end_idx = -1

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
            # ranks.append(ranks_[j])
            # name_tags.append(name_tags_[j])
            positions.append(positions_[j])
            spaces.append(spaces_[j])

    if start_idx == -1 or end_idx == -1:
        continue

    #print(answer_text, tokens[start_idx: end_idx + 1])

    ids = tokenization.convert_tokens_to_ids(vocab=vocab, tokens=tokens)

    length = len(ids)
    if length > max_length:
        length = max_length

    for j in range(length):
        sequence_has_ans[count, 0, j] = ids[j]
        sequence_has_ans[count, 1, j] = segments[j]
        sequence_has_ans[count, 2, j] = cols[j]
        sequence_has_ans[count, 3, j] = rows[j]
    answer_text_array[count] = answer_text
    answer_span[count, 0] = start_idx
    answer_span[count, 1] = end_idx
    question_text_array[count] = questions[i]
    count += 1

sequence_has_ans_ = np.zeros(shape=[count, 4, max_length], dtype=np.int32)
answer_span_ = np.zeros(shape=[count, 2], dtype=np.int32)
answer_text_array_ = np.zeros(shape=[count], dtype='<U200')
question_text_array_ = np.zeros(shape=[count], dtype='<U200')

for i in range(count):
    sequence_has_ans_[i] = sequence_has_ans[i]
    answer_span_[i] = answer_span[i]
    answer_text_array_[i] = answer_text_array[i]
    question_text_array_[i] = question_text_array[i]

np.save('sequence_table_factory', sequence_has_ans_)
np.save('answer_span_factory', answer_span_)
np.save('answer_text_array_factory', answer_text_array_)
np.save('question_text_array_factory', question_text_array_)


print(count, false_count, file_error)