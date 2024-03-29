import numpy as np
import random

class DataHolder:
    def __init__(self):
        """
        self.input_ids2 = np.load('sequence_table_testset.npy')
        self.input_mask2 = np.load('mask_table_testset.npy')
        self.input_segments2 = np.load('segments_table_testset.npy')
        self.input_rows2 = np.load('rows_table_testset.npy')
        self.input_cols2 = np.load('cols_table_testset.npy')
        self.answer_span2 = np.load('answer_testset.npy')

        self.input_ids3 = np.load('sequence_table_testset2.npy')
        self.input_mask3 = np.load('mask_table_testset2.npy')
        self.input_segments3 = np.load('segments_table_testset2.npy')
        self.input_rows3 = np.load('rows_table_testset2.npy')
        self.input_cols3 = np.load('cols_table_testset2.npy')
        self.answer_span3 = np.load('answer_testset2.npy')
        """
        #
        # self.input_ids_test = np.load('input_ids_test.npy')
        # self.input_mask_test = np.load('input_masks_test.npy')
        # self.input_segments_test = np.load('input_segments_test.npy')
        # self.input_positions_test = np.load('positions_table.npy')
        # self.input_rows_test = np.load('input_rows_test.npy')
        # self.input_cols_test = np.load('input_cols_test.npy')
        # self.answer_span_test = np.load('answer_text_test.npy')

        self.input_ids = np.load('sequence_table.npy')
        self.input_mask = np.load('mask_table.npy')
        self.input_segments = np.load('segments_table.npy')
        self.input_rows = np.load('rows_table.npy')
        self.input_cols = np.load('cols_table.npy')
        self.answer_span = np.load('answer_span_table.npy')
        print('KorQuAD 2.0 ', self.input_ids.shape)

        self.input_ids2 = np.load('sequence_table2.npy')
        self.input_mask2 = np.load('mask_table2.npy')
        self.input_segments2 = np.load('segments_table2.npy')
        self.input_rows2 = np.load('rows_table2.npy')
        self.input_cols2 = np.load('cols_table2.npy')
        self.answer_span2 = np.load('answer_span_table2.npy').astype(dtype=np.int32)
        self.answer_texts2 = np.load('answer_texts2.npy')
        print('KorWIKI ',self.input_ids2.shape)

        self.input_ids2_ = np.load('sequence_table2_.npy')
        self.input_segments2_ = np.load('segments_table2_.npy')
        self.input_rows2_ = np.load('rows_table2_.npy')
        self.input_cols2_ = np.load('cols_table2_.npy')
        self.answer_texts2_ = np.load('answer_texts2_.npy')
        self.answer_span2_ = np.load('answer_span2_test.npy')
        print('KorWIKI test,,?: ', self.input_ids2_.shape)

        self.input_text = np.load('sequence_crs.npy')
        self.segments_text = np.load('segments_crs.npy')
        self.mask_text = np.load('mask_crs.npy')
        self.rows_text = np.load('rows_crs.npy')
        self.answer_span_text = np.load('answer_span_crs.npy')

        # office QA Dataset
        # """
        self.input_ids3_ = np.load('trainset/input_ids.npy')
        self.input_segments3_ = np.load('trainset/input_segment.npy')
        self.input_rows3_ = np.load('trainset/input_row.npy')
        self.input_cols3_ = np.load('trainset/input_col.npy')
        self.answer_span3_ = np.load('trainset/answer_span.npy')
        print('office train', self.input_ids3_.shape)
        # """

        self.input_ids3 = np.load('testset/input_ids.npy')
        self.input_segments3 = np.load('testset/input_segment.npy')
        self.input_rows3 = np.load('testset/input_row.npy')
        self.input_cols3 = np.load('testset/input_col.npy')
        self.answer_span3 = np.load('testset/answer_span.npy')
        self.answer_lists3 = np.load('testset/answer_text_list.npy')
        print('office_test: ', self.input_ids3.shape)

        # spec table Dataset
        self.input_ids4 = np.load('sequence_table_factory.npy')
        self.answer_span4 = np.load('answer_span_factory.npy')
        self.answer_lists4 = np.load('answer_text_array_factory.npy')
        print('spec ', self.input_ids4.shape)

        # law QA
        self.input_ids5 = np.load('input_ids_law.npy')
        self.answer_span5 = np.load('answer_span_law.npy')
        self.answer_lists5 = np.load('answer_list_law.npy')
        print('law ', self.input_ids5.shape)

        # # aihub QA
        # self.input_ids_hub = np.load('./np/sequence_table_aihub.npy')
        # self.input_segment_hub = np.load('./np/segments_table_aihub.npy')
        # self.input_mask_hub = np.load('./np/mask_table_aihub.npy')
        # self.input_row_hub = np.load('./np/rows_table_aihub.npy')
        # self.input_col_hub = np.load('./np/cols_table_aihub.npy')
        # self.answer_span_hub = np.load('./np/answer_span_table_aihub.npy')
        # self.answer_text_hub = np.load('./np/answer_texts_aihub.npy')
        # print('input shape:', self.input_ids_hub.shape)

        # # dense-sparse 용임
        # self.p_ids = np.load('p_ids.npy')
        # self.q_ids = np.load('q_ids.npy')

        # print(self.input_ids2.shape, self.input_ids3.shape)

        ## random shuffle
        # KorQuAD
        self.r_ix = np.array(range(self.input_ids.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix)

        # KorWIKI
        self.r_ix1 = np.array(range(self.input_ids2.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix1)
        self.r_ix1 = self.r_ix1[:10000]

        #
        self.r_ix2 = np.array(range(self.input_text.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix2)

        # wii test
        self.r_ix3 = np.array(range(self.input_ids2_.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix3)

        # office
        self.r_ix3_ = np.array(range(self.input_ids3_.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix3_)

        # spec
        self.r_ix4 = np.array(range(self.input_ids4.shape[0] - 300), dtype=np.int32)
        np.random.shuffle(self.r_ix4)

        # law
        self.r_ix5 = np.array(range(self.input_ids5.shape[0] - 300), dtype=np.int32)
        np.random.shuffle(self.r_ix5)

        # self.r_ix_r = np.array(range(self.p_ids.shape[0]), dtype=np.int32)
        # np.random.shuffle(self.r_ix_r)

        self.batch_size = 1
        self.b_ix = 0
        self.b_ix1 = 0
        self.b_ix2 = 0
        self.b_ix3 = 0
        self.b_ix4 = 0
        self.b_ix5 = 0

        self.b_ix_r = 0
        self.step = 0

        self.next_domain = 0
        self.ep = 0 # 1
        self.ep2 = 0 # 1
        self.ep3 = 0 # 2
        self.ep4 = 0 # 5
        self.ep5 = 0 # 6

    # KorQuAD
    def next_batch(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0
        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0
        if self.b_ix3 + self.batch_size * 2 > self.input_ids3.shape[0]:
            self.b_ix3 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_positions = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        start_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        for i in range(self.batch_size):
            ix = self.r_ix[self.b_ix]

            try:
                start_label[i, self.answer_span[ix, 0]] = 1
                stop_label[i, self.answer_span[ix, 1]] = 1
            except:
                self.b_ix += 1
                return self.next_batch()

            input_ids[i] = self.input_ids[ix, 0]
            input_segments[i] = self.input_segments[ix, 0]
            input_mask[i] = self.input_mask[ix, 0]
            input_rows[i] = self.input_rows[ix, 0]
            input_cols[i] = self.input_cols[ix, 0]

            self.b_ix += 1

        return input_ids, input_mask, input_segments, input_rows, input_cols, start_label, stop_label

    def next_batch_korquad_tqa(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix1 + self.batch_size * 2 > self.input_ids2.shape[0]:
            self.b_ix1 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)

        start_label = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)

        for i in range(self.batch_size):
            if self.step % 3 == 0:
                ix = self.r_ix[self.b_ix]

                try:
                    start_label[i, self.answer_span[ix, 0]] = 1
                    stop_label[i, self.answer_span[ix, 1]] = 1
                except:
                    self.b_ix += 1
                    return self.next_batch()

                input_ids[i] = self.input_ids[ix, 0]
                input_segments[i] = self.input_segments[ix, 0]
                input_mask[i] = self.input_mask[ix, 0]
                input_rows[i] = self.input_rows[ix, 0]
                input_cols[i] = self.input_cols[ix, 0]

                self.b_ix += 1
            else:
                ix = self.r_ix1[self.b_ix1]

                try:
                    start_label[i, self.answer_span2[ix, 0]] = 1
                    stop_label[i, self.answer_span2[ix, 1]] = 1
                except:
                    self.b_ix1 += 1
                    return self.next_batch_tqa()

                input_ids[i] = self.input_ids2[ix]
                input_segments[i] = self.input_segments2[ix]
                input_rows[i] = self.input_rows2[ix]
                input_cols[i] = self.input_cols2[ix]

                self.b_ix1 += 1
            self.step += 1

        return input_ids, input_mask, input_segments, input_rows, input_cols, start_label, stop_label

    # KorWIKI
    def next_batch_tqa(self):
        input_ids = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)

        start_label = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)

        for i in range(self.batch_size):
            try:
                ix = self.r_ix1[self.b_ix1]
            except:
                self.b_ix1 = 0
                ix = self.r_ix1[self.b_ix1]

            try:
                start_label[i, self.answer_span2[ix, 0]] = 1
                stop_label[i, self.answer_span2[ix, 1]] = 1
            except:
                self.b_ix1 += 1
                return self.next_batch_tqa()

            input_ids[i] = self.input_ids2[ix]
            input_segments[i] = self.input_segments2[ix]
            input_rows[i] = self.input_rows2[ix]
            input_cols[i] = self.input_cols2[ix]

            self.b_ix1 += 1

        return input_ids, input_mask, input_segments, input_rows, input_cols, start_label, stop_label

    def test_batch_tqa(self):
        if self.b_ix1 + self.batch_size * 2 + 300 > self.input_ids2_.shape[0]:
            self.b_ix1 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids2_.shape[1]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids2_.shape[1]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids2_.shape[1]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids2_.shape[1]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids2_.shape[1]], dtype=np.int32)
        answer_texts = np.zeros(shape=[self.batch_size, self.input_ids2_.shape[1]], dtype='<U50')

        for i in range(self.batch_size):
            ix = self.r_ix3[self.b_ix1]

            input_ids[i] = self.input_ids2_[ix]
            input_segments[i] = self.input_segments2_[ix]
            input_rows[i] = self.input_rows2_[ix]
            input_cols[i] = self.input_cols2_[ix]
            answer_texts[i] = self.answer_texts2_[ix]

            self.b_ix1 += 1

        return input_ids, input_mask, input_segments, input_rows, input_cols, answer_texts

    # OFFICE, SPEC, LAW
    def next_batch_office(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        if self.b_ix3 + self.batch_size * 2 + 300 > self.input_ids2.shape[0]:
            self.b_ix3 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_positions = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        start_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        for i in range(self.batch_size):
            ix = self.r_ix3_[self.b_ix3]

            try:
                #print(self.answer_span2[ix])
                start_label[i, int(self.answer_span3_[ix, 0])] = 1
                stop_label[i, int(self.answer_span3_[ix, 1])] = 1
            except:
                self.b_ix3 += 1
                return self.next_batch_office()

            input_ids[i] = self.input_ids3_[ix]
            input_segments[i] = self.input_segments3_[ix]
            input_rows[i] = self.input_rows3_[ix]
            input_cols[i] = self.input_cols3_[ix]
            self.b_ix3 += 1

        return input_ids, input_mask, input_segments, input_rows, input_cols, start_label, stop_label

    def next_batch_spec(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        if self.b_ix4 + self.batch_size * 2 > self.input_ids4.shape[0] - 300:
            self.b_ix4 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_positions = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        start_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        for i in range(self.batch_size):
            ix = self.r_ix4[self.b_ix4]

            try:
                #print(self.answer_span2[ix])
                start_label[i, int(self.answer_span4[ix, 0])] = 1
                stop_label[i, int(self.answer_span4[ix, 1])] = 1
            except:
                self.b_ix4 += 1
                return self.next_batch_spec()

            input_ids[i] = self.input_ids4[ix, 0]
            input_segments[i] = self.input_ids4[ix, 1]
            input_rows[i] = self.input_ids4[ix, 3]
            input_cols[i] = self.input_ids4[ix, 2]

            self.b_ix4 += 1

        return input_ids, input_mask, input_segments, input_rows, input_cols, start_label, stop_label

    def next_batch_law(self):
        if self.b_ix5 + self.batch_size * 2 > self.input_ids5.shape[0] - 300:
            self.b_ix5 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_positions = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        start_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        for i in range(self.batch_size):
            ix = self.r_ix5[self.b_ix5]

            try:
                #print(self.answer_span2[ix])
                start_label[i, int(self.answer_span5[ix, 0])] = 1
                stop_label[i, int(self.answer_span5[ix, 1])] = 1
            except:
                self.b_ix5 += 1
                return self.next_batch_law()

            input_ids[i] = self.input_ids5[ix, 0]
            input_segments[i] = self.input_ids5[ix, 1]
            input_rows[i] = self.input_ids5[ix, 2]
            input_cols[i] = self.input_ids5[ix, 3]

            self.b_ix5 += 1

        return input_ids, input_mask, input_segments, input_rows, input_cols, start_label, stop_label

    def test_batch_office(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        if self.b_ix3 + self.batch_size * 2 > self.input_ids2.shape[0]:
            self.b_ix3 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        answer_text = np.zeros(shape=[self.batch_size], dtype='<U50')

        for i in range(self.batch_size):
            ix = self.b_ix4

            input_ids[i] = self.input_ids3[ix]
            input_segments[i] = self.input_segments3[ix]
            input_rows[i] = self.input_rows3[ix]
            input_cols[i] = self.input_cols3[ix]
            answer_text[i] = self.answer_lists3[ix]

            self.b_ix4 += 1

        return input_ids, input_segments, input_rows, input_cols, answer_text

    # 300
    def test_batch_spec(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        if self.b_ix3 + self.batch_size * 2 > self.input_ids2.shape[0]:
            self.b_ix3 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        answer_text = np.zeros(shape=[self.batch_size], dtype='<U50')

        for i in range(self.batch_size):
            ix = self.input_ids4.shape[0] - 300 + self.b_ix4
            input_ids[i] = self.input_ids4[ix, 0]
            input_segments[i] = self.input_ids4[ix, 1]
            input_rows[i] = self.input_ids4[ix, 3]
            input_cols[i] = self.input_ids4[ix, 2]
            answer_text[i] = self.answer_lists4[ix]

            self.b_ix4 += 1

        return input_ids, input_segments, input_rows, input_cols, answer_text

    # 300
    def test_batch_law(self):
        if self.b_ix5 + self.batch_size * 2 > self.input_ids5.shape[0]:
            self.b_ix5 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids5.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids5.shape[2]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids5.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids5.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids5.shape[2]], dtype=np.int32)
        answer_text = np.zeros(shape=[self.batch_size], dtype='<U50')

        for i in range(self.batch_size):
            ix = self.input_ids5.shape[0] - 300 + self.b_ix5

            input_ids[i] = self.input_ids5[ix, 0]
            input_segments[i] = self.input_ids5[ix, 1]
            input_rows[i] = self.input_ids5[ix, 2]
            input_cols[i] = self.input_ids5[ix, 3]
            answer_text[i] = self.answer_lists5[ix]

            self.b_ix5 += 1

        return input_ids, input_segments, input_rows, input_cols, answer_text

    def test_batch_tqa(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix1 + self.batch_size * 2 + 300 > self.input_ids2.shape[0]:
            self.b_ix1 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        answer_text = np.zeros(shape=[self.batch_size], dtype='<U50')

        for i in range(self.batch_size):
            ix = self.b_ix1

            input_ids[i] = self.input_ids2_[ix]
            input_segments[i] = self.input_segments2_[ix]
            input_rows[i] = self.input_rows2_[ix]
            input_cols[i] = self.input_cols2_[ix]
            answer_text[i] = self.answer_texts2_[ix]

            self.b_ix1 += 1

        return input_ids, input_segments, input_rows, input_cols, answer_text

    def next_batch_all(self):
        # 확률 생성
        prob = random.uniform(0, 1)
        all_size = self.input_ids.shape[0] + self.input_ids2.shape[0] + self.input_ids3_.shape[0] \
                   + self.input_ids4.shape[0] + self.input_ids5.shape[0]
        # print(prob, all_size)
        # KorQuAD
        if prob <= self.input_ids.shape[0] / all_size:
            return self.next_batch_korquad_tqa()
        # WIkiTQ
        elif prob <= self.input_ids.shape[0] + self.input_ids2.shape[0] / all_size:
            return self.next_batch_tqa()
        # Office
        elif prob <= self.input_ids.shape[0] + self.input_ids2.shape[0] + self.input_ids3_.shape[0] / all_size:
            return self.next_batch_office()
        # Spec
        elif prob <= self.input_ids.shape[0] + self.input_ids2.shape[0] + self.input_ids3_.shape[0]\
                + self.input_ids4.shape[0] / all_size:
            return self.next_batch_spec()
        # Law
        elif prob <= 1 / all_size:
            return self.next_batch_law()

    def next_batch_all_adv(self):
        domain_num = 5
        self.batch_size = 4

        # domain
        domain_label = np.zeros(shape=[self.batch_size, domain_num], dtype=np.float32)
        # 확률 생성
        all_size = self.input_ids.shape[0] + self.input_ids2.shape[0] + self.input_ids3_.shape[0] \
                   + self.input_ids4.shape[0] + self.input_ids5.shape[0]

        # KorQuAD - ids
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0
        # Korwiki - ids2
        if self.b_ix2 + self.batch_size * 2 + 300 > self.input_ids2.shape[0]:
            self.b_ix2= 0
        # offICE - ids3_
        if self.b_ix3 + self.batch_size * 2 + 300 > self.input_ids3_.shape[0]:
            self.b_ix3 = 0
        # SPEC - ids4
        if self.b_ix4 + self.batch_size * 2 > self.input_ids4.shape[0] - 300:
            self.b_ix4 = 0
        # LAW - ids5
        if self.b_ix5 + self.batch_size * 2 > self.input_ids5.shape[0] - 300:
            self.b_ix5 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        start_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        for i in range(self.batch_size):
            prob = random.uniform(0, 1)

            if prob <= self.input_ids.shape[0] / all_size:
                ix = self.r_ix[self.b_ix]

                try:
                    start_label[i, self.answer_span[ix, 0]] = 1
                    stop_label[i, self.answer_span[ix, 1]] = 1
                except:
                    self.b_ix += 1
                    return self.next_batch_all_adv()

                input_ids[i] = self.input_ids[ix, 0]
                input_segments[i] = self.input_segments[ix, 0]
                input_mask[i] = self.input_mask[ix, 0]
                input_rows[i] = self.input_rows[ix, 0]
                input_cols[i] = self.input_cols[ix, 0]
                domain_label[i, 0] = 1

                self.b_ix += 1

            elif prob <= self.input_ids.shape[0] + self.input_ids2.shape[0] / all_size:
                if self.b_ix2 > self.input_ids.shape[0]:
                    self.b_ix2 = 0
                ix = self.r_ix1[self.b_ix2]

                try:
                    start_label[i, self.answer_span2[ix, 0]] = 1
                    stop_label[i, self.answer_span2[ix, 1]] = 1
                except:
                    self.b_ix2 += 1
                    return self.next_batch_all_adv()

                input_ids[i] = self.input_ids2[ix]
                input_segments[i] = self.input_segments2[ix]
                input_rows[i] = self.input_rows2[ix]
                input_cols[i] = self.input_cols2[ix]
                domain_label[i, 1] = 1

                self.b_ix2 += 1

            elif prob <= self.input_ids.shape[0] + self.input_ids2.shape[0] + self.input_ids3_.shape[0] / all_size:
                ix = self.r_ix3_[self.b_ix3]

                try:
                    # print(self.answer_span2[ix])
                    start_label[i, int(self.answer_span3_[ix, 0])] = 1
                    stop_label[i, int(self.answer_span3_[ix, 1])] = 1
                except:
                    self.b_ix3 += 1
                    return self.next_batch_all_adv()

                input_ids[i] = self.input_ids3_[ix]
                input_segments[i] = self.input_segments3_[ix]
                input_rows[i] = self.input_rows3_[ix]
                input_cols[i] = self.input_cols3_[ix]
                domain_label[i, 2] = 1

                self.b_ix3 += 1

            elif prob <= self.input_ids.shape[0] + self.input_ids2.shape[0] + self.input_ids3_.shape[0] + \
                self.input_ids4.shape[0] / all_size:

                ix = self.r_ix4[self.b_ix4]

                try:
                    start_label[i, int(self.answer_span4[ix, 0])] = 1
                    stop_label[i, int(self.answer_span4[ix, 1])] = 1
                except:
                    self.b_ix4 += 1
                    return self.next_batch_all_adv()

                input_ids[i] = self.input_ids4[ix, 0]
                input_segments[i] = self.input_ids4[ix, 1]
                input_rows[i] = self.input_ids4[ix, 3]
                input_cols[i] = self.input_ids4[ix, 2]
                domain_label[i, 3] = 1

                self.b_ix4 += 1
            else:
                ix = self.r_ix5[self.b_ix5]

                try:
                    # print(self.answer_span2[ix])
                    start_label[i, int(self.answer_span5[ix, 0])] = 1
                    stop_label[i, int(self.answer_span5[ix, 1])] = 1
                except:
                    self.b_ix5 += 1
                    return self.next_batch_all_adv()

                input_ids[i] = self.input_ids5[ix, 0]
                input_segments[i] = self.input_ids5[ix, 1]
                input_rows[i] = self.input_ids5[ix, 2]
                input_cols[i] = self.input_ids5[ix, 3]
                domain_label[i, 4] = 1

                self.b_ix5 += 1

        return input_ids, input_mask, input_segments, input_rows, input_cols, start_label, stop_label, domain_label

    def next_batch_all_adv2(self):
        domain_num = 5
        self.batch_size = 4

        # domain
        domain_label = np.zeros(shape=[self.batch_size, domain_num], dtype=np.float32)
        input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        start_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        ## KorQuAD - ids
        if self.b_ix + self.batch_size > self.input_ids.shape[0]:
            self.b_ix = 0
        # Korwiki - ids2
        if self.b_ix2 + self.batch_size > 10000:
            self.b_ix2 = 0
        # offICE - ids3_
        if self.b_ix3 + self.batch_size > self.input_ids3_.shape[0]:
            self.b_ix3 = 0
        # SPEC - ids4
        if self.b_ix4 + self.batch_size > self.input_ids4.shape[0] - 300:
            self.b_ix4 = 0
        # LAW - ids5
        if self.b_ix5 + self.batch_size > self.input_ids5.shape[0] - 300:
            self.b_ix5 = 0

        i = 0
        while i != self.batch_size:
            # Domain check
            if self.next_domain == domain_num:
                self.next_domain = 0

            if self.next_domain == 0:
                idx = self.r_ix[self.b_ix]

                try:
                    start_label[i, self.answer_span[idx, 0]] = 1
                    stop_label[i, self.answer_span[idx, 1]] = 1
                except:
                    self.b_ix += 1
                    continue

                input_ids[i] = self.input_ids[idx, 0]
                input_segments[i] = self.input_segments[idx, 0]
                input_mask[i] = self.input_mask[idx, 0]
                input_rows[i] = self.input_rows[idx, 0]
                input_cols[i] = self.input_cols[idx, 0]
                domain_label[i, 0] = 1

                self.b_ix += 1
                self.next_domain += 1
                i += 1

            elif self.next_domain == 1:
                idx = self.r_ix1[self.b_ix2]

                try:
                    start_label[i, self.answer_span2[idx, 0]] = 1
                    stop_label[i, self.answer_span2[idx, 1]] = 1
                except:
                    self.b_ix2 += 1
                    continue

                input_ids[i] = self.input_ids2[idx]
                input_segments[i] = self.input_segments2[idx]
                input_mask[i] = self.input_mask2[idx, 0]
                input_rows[i] = self.input_rows2[idx]
                input_cols[i] = self.input_cols2[idx]
                domain_label[i, 1] = 1

                self.b_ix2 += 1
                self.next_domain += 1
                i += 1

            elif self.next_domain == 2:
                idx = self.r_ix3_[self.b_ix3]

                try:
                    start_label[i, int(self.answer_span3_[idx, 0])] = 1
                    stop_label[i, int(self.answer_span3_[idx, 1])] = 1
                except:
                    self.b_ix3 += 1
                    continue

                input_ids[i] = self.input_ids3_[idx]
                input_segments[i] = self.input_segments3_[idx]
                input_rows[i] = self.input_rows3_[idx]
                input_cols[i] = self.input_cols3_[idx]
                domain_label[i, 2] = 1

                self.b_ix3 += 1
                self.next_domain += 1
                i += 1
            elif self.next_domain == 3:
                idx = self.r_ix4[self.b_ix4]

                try:
                    start_label[i, int(self.answer_span4[idx, 0])] = 1
                    stop_label[i, int(self.answer_span4[idx, 1])] = 1
                except:
                    self.b_ix4 += 1
                    continue

                input_ids[i] = self.input_ids4[idx, 0]
                input_segments[i] = self.input_ids4[idx, 1]
                input_rows[i] = self.input_ids4[idx, 3]
                input_cols[i] = self.input_ids4[idx, 2]
                domain_label[i, 3] = 1

                self.b_ix4 += 1
                self.next_domain += 1
                i += 1

            else:
                idx = self.r_ix5[self.b_ix5]

                try:
                    start_label[i, int(self.answer_span5[idx, 0])] = 1
                    stop_label[i, int(self.answer_span5[idx, 1])] = 1
                except:
                    self.b_ix5 += 1
                    continue

                input_ids[i] = self.input_ids5[idx, 0]
                input_segments[i] = self.input_ids5[idx, 1]
                input_rows[i] = self.input_ids5[idx, 3]
                input_cols[i] = self.input_ids5[idx, 2]
                domain_label[i, 4] = 1

                self.b_ix5 += 1
                self.next_domain += 1
                i += 1
        #domain_label[:] = 0
        return input_ids, input_mask, input_segments, input_rows, input_cols, start_label, stop_label, domain_label

# DataHolder = DataHolder()
