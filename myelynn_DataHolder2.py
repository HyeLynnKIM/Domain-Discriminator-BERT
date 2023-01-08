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
        # self.input_ids_test = np.load('./np/input_ids_test.npy')
        # self.input_mask_test = np.load('./np/input_masks_test.npy')
        # self.input_segments_test = np.load('./np/input_segments_test.npy')
        # self.input_positions_test = np.load('./np/positions_table.npy')
        # self.input_rows_test = np.load('./np/input_rows_test.npy')
        # self.input_cols_test = np.load('./np/input_cols_test.npy')
        # self.answer_span_test = np.load('./np/answer_text_test.npy')

        # KorQuAD 2.0 Table
        self.input_ids = np.load('./np/sequence_table.npy')
        self.input_mask = np.load('./np/mask_table.npy')
        self.input_segments = np.load('./np/segments_table.npy')
        self.input_rows = np.load('./np/rows_table.npy')
        self.input_cols = np.load('./np/cols_table.npy')
        self.answer_span = np.load('./np/answer_span_table.npy')
        print('input shape:', self.input_ids.shape)

        # KorWikiTQ
        self.input_ids2 = np.load('./np/sequence_table2.npy')
        self.input_mask2 = np.load('./np/mask_table2.npy')
        self.input_segments2 = np.load('./np/segments_table2.npy')
        self.input_rows2 = np.load('./np/rows_table2.npy')
        self.input_cols2 = np.load('./np/cols_table2.npy')
        self.answer_span2 = np.load('./np/answer_span_table2.npy').astype(dtype=np.int32)
        self.answer_texts2 = np.load('./np/answer_texts2.npy')
        print('input shape:',self.input_ids2.shape)

        # KorWikiTQ dev
        self.input_ids2_ = np.load('./np/sequence_table2_.npy')
        self.input_segments2_ = np.load('./np/segments_table2_.npy')
        self.input_rows2_ = np.load('./np/rows_table2_.npy')
        self.input_cols2_ = np.load('./np/cols_table2_.npy')
        self.answer_texts2_ = np.load('./np/answer_texts2_.npy')
        print('input shape:', self.input_ids2_.shape)

        # KorQuAD 2.0 text
        self.input_text = np.load('./np/sequence_crs.npy')
        self.segments_text = np.load('./np/segments_crs.npy')
        self.mask_text = np.load('./np/mask_crs.npy')
        self.rows_text = np.load('./np/rows_crs.npy')
        self.answer_span_text = np.load('./np/answer_span_crs.npy')
        print('input shape:', self.input_text.shape)
        #"""

        # office QA Dataset Train set
        self.input_ids3_ = np.load('./np/trainset/input_ids.npy')
        self.input_segments3_ = np.load('./np/trainset/input_segment.npy')
        self.input_rows3_ = np.load('./np/trainset/input_row.npy')
        self.input_cols3_ = np.load('./np/trainset/input_col.npy')
        self.answer_span3_ = np.load('./np/trainset/answer_span.npy')
        print('input shape:', self.input_ids3_.shape)
        #"""

        # office QA Dataset Test set
        self.input_ids3 = np.load('./np/testset/input_ids.npy')
        self.input_segments3 = np.load('./np/testset/input_segment.npy')
        self.input_rows3 = np.load('./np/testset/input_row.npy')
        self.input_cols3 = np.load('./np/testset/input_col.npy')
        self.answer_span3 = np.load('./np/testset/answer_span.npy')
        self.answer_lists3 = np.load('./np/testset/answer_text_list.npy')
        print('input shape:', self.input_ids3.shape)

        # spec table Dataset
        self.input_ids4 = np.load('./np/sequence_table_factory.npy')
        self.answer_span4 = np.load('./np/answer_span_factory.npy')
        self.answer_lists4 = np.load('./np/answer_text_array_factory.npy')
        print('input shape:', self.input_ids4.shape)

        # law QA
        self.input_ids5 = np.load('./np/input_ids_law.npy')
        self.answer_span5 = np.load('./np/answer_span_law.npy')
        self.answer_lists5 = np.load('./np/answer_list_law.npy')
        print('input shape:', self.input_ids5.shape)

        # aihub QA
        self.input_ids_hub = np.load('./np/sequence_table_aihub.npy')
        self.input_segment_hub = np.load('./np/segments_table_aihub.npy')
        self.input_mask_hub = np.load('./np/mask_table_aihub.npy')
        self.input_row_hub = np.load('./np/rows_table_aihub.npy')
        self.input_col_hub = np.load('./np/cols_table_aihub.npy')
        self.answer_span_hub = np.load('./np/answer_span_table_aihub.npy')
        self.answer_text_hub = np.load('./np/answer_texts_aihub.npy')
        print('input shape:', self.input_ids_hub.shape)

        # aihub QA validation
        self.input_ids_hub_ = np.load('./np/VL_sequence_table_aihub.npy')
        self.input_segment_hub_ = np.load('./np/VL_segments_table_aihub.npy')
        self.input_mask_hub_ = np.load('./np/VL_mask_table_aihub.npy')
        self.input_row_hub_ = np.load('./np/VL_rows_table_aihub.npy')
        self.input_col_hub_ = np.load('./np/VL_cols_table_aihub.npy')
        self.answer_span_hub_ = np.load('./np/VL_answer_span_table_aihub.npy')
        self.answer_text_hub_ = np.load('./np/VL_answer_texts_aihub.npy')
        print('input shape:', self.input_ids_hub_.shape)

        # # dense-sparse 용임
        # self.p_ids = np.load('p_ids.npy')
        # self.q_ids = np.load('q_ids.npy')

        print(self.input_ids2.shape, self.input_ids3.shape)

        ## random shuffle
        self.r_ix = np.array(range(self.input_ids.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix)

        self.r_ix1 = np.array(range(self.input_ids2.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix1)

        self.r_ix2 = np.array(range(self.input_text.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix2)

        self.r_ix3 = np.array(range(self.input_ids2.shape[0]), dtype=np.int32)
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

        self.batch_size = 4
        self.b_ix = 0
        self.b_ix1 = 0
        self.b_ix2 = 0
        self.b_ix3 = 0
        self.b_ix4 = 0
        self.b_ix5 = 0

        self.b_ix_r = 0

        self.step = 0

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

    def next_batch_tqa(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix1 + self.batch_size * 2 + 300 > self.input_ids2.shape[0]:
            self.b_ix1 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)

        start_label = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)

        for i in range(self.batch_size):
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

    def next_batch_combine(self):
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

    def next_batch3(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        if self.b_ix3 + self.batch_size * 2 > self.input_ids2.shape[0]:
            self.b_ix3 = 0

        if self.b_ix4 + self.batch_size * 2 > self.input_ids4.shape[0] - 300:
            self.b_ix4 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        start_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        for i in range(self.batch_size - 1):
            ix = self.r_ix4[self.b_ix4]

            try:
                #print(self.answer_span2[ix])
                start_label[i, int(self.answer_span4[ix, 0])] = 1
                stop_label[i, int(self.answer_span4[ix, 1])] = 1
            except:
                self.b_ix4 += 1
                return self.next_batch3()

            input_ids[i] = self.input_ids4[ix, 0]
            input_segments[i] = self.input_ids4[ix, 1]
            input_rows[i] = self.input_ids4[ix, 3]
            input_cols[i] = self.input_ids4[ix, 2]

            self.b_ix4 += 1

        if self.step % 2 == 0:
            ix = self.r_ix3[self.b_ix3]

            try:
                # print(self.answer_span2[ix])
                start_label[self.batch_size - 1, int(self.answer_span2[ix, 0])] = 1
                stop_label[self.batch_size - 1, int(self.answer_span2[ix, 1])] = 1
            except:
                self.b_ix3 += 1
                return self.next_batch_combine()

            input_ids[self.batch_size - 1] = self.input_ids2[ix]
            input_segments[self.batch_size - 1] = self.input_segments2[ix]
            input_rows[self.batch_size - 1] = self.input_rows2[ix]
            input_cols[self.batch_size - 1] = self.input_cols2[ix]

            self.b_ix3 += 1
        else:
            ix = self.r_ix4[self.b_ix4]
            try:
                # print(self.answer_span2[ix])
                start_label[self.batch_size - 1, int(self.answer_span4[ix, 0])] = 1
                stop_label[self.batch_size - 1, int(self.answer_span4[ix, 1])] = 1
            except:
                self.b_ix3 += 1
                return self.next_batch_combine()

            input_ids[self.batch_size - 1] = self.input_ids4[ix, 0]
            input_segments[self.batch_size - 1] = self.input_ids4[ix, 1]
            input_rows[self.batch_size - 1] = self.input_ids4[ix, 3]
            input_cols[self.batch_size - 1] = self.input_ids4[ix, 2]

            self.b_ix4 += 1

        self.step += 1

        return input_ids, input_mask, input_segments, input_rows, input_cols, start_label, stop_label

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
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

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

    def test_batch_law(self):
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
            ix = self.input_ids5.shape[0] - 300 + self.b_ix5

            input_ids[i] = self.input_ids5[ix, 0]
            input_segments[i] = self.input_ids5[ix, 1]
            input_rows[i] = self.input_ids5[ix, 2]
            input_cols[i] = self.input_ids5[ix, 3]
            answer_text[i] = self.answer_lists5[ix]

            self.b_ix5 += 1

        return input_ids, input_segments, input_rows, input_cols, answer_text

    def test_batch_hub(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix1 + self.batch_size * 2 + 300 > self.input_ids2.shape[0]:
            self.b_ix1 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids_hub_.shape[1]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_segment_hub_.shape[1]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_row_hub_.shape[1]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_col_hub_.shape[1]], dtype=np.int32)
        answer_text = np.zeros(shape=[self.batch_size], dtype='<U50')

        for i in range(self.batch_size):
            ix = self.b_ix1
            input_ids[i] = self.input_ids_hub_[ix]
            input_segments[i] = self.input_segment_hub_[ix]
            input_rows[i] = self.input_row_hub_[ix]
            input_cols[i] = self.input_col_hub_[ix]
            answer_text[i] = self.answer_text_hub_[ix]

            self.b_ix1 += 1

        return input_ids, input_segments, input_rows, input_cols, answer_text

    # def next_batch_test(self):
    #     self.batch_size = 1
    #
    #     if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
    #         self.b_ix = 0
    #
    #     if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
    #         self.b_ix2 = 0
    #
    #     input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
    #     input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
    #     input_positions = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
    #     input_mask = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
    #     input_rows = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
    #     input_cols = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
    #     input_names = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
    #     input_rankings = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
    #
    #     answer_texts = np.zeros(shape=[self.batch_size], dtype=np.int32)
    #
    #     for i in range(self.batch_size):
    #         ix = self.b_ix
    #
    #         input_ids[i] = self.input_ids_test[ix]
    #         input_segments[i] = self.input_segments_test[ix]
    #         #input_positions[i] = self.input_positions_test[ix]
    #         input_mask[i] = self.input_mask_test[ix]
    #         input_rows[i] = self.input_rows_test[ix]
    #         input_cols[i] = self.input_cols_test[ix]
    #         text = self.answer_span_test[ix]
    #
    #         self.b_ix += 1
    #
    #     return input_ids, input_mask, input_segments, input_rows, \
    #            input_cols, text

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
        # domain
        domain_label = np.zeros(shape=[self.batch_size, 5], dtype=np.float32)

        # 확률 생성
        prob = random.uniform(0, 1)
        all_size = self.input_ids.shape[0] + self.input_ids2.shape[0] + self.input_ids3_.shape[0] \
                   + self.input_ids4.shape[0] + self.input_ids5.shape[0]

        # KorQuAD
        if prob <= self.input_ids.shape[0] / all_size:
            input_, mask, seg, r, c, stal, stpl = self.next_batch_korquad_tqa()
            for i in range(self.batch_size):
                domain_label[i, 0] = 1
            return input_, mask, seg, r, c, stal, stpl, domain_label
        # WIkiTQ
        elif prob <= (self.input_ids.shape[0] + self.input_ids2.shape[0]) / all_size:
            input_, mask, seg, r, c, stal, stpl = self.next_batch_tqa()
            for i in range(self.batch_size):
                domain_label[i, 1] = 1
            return input_, mask, seg, r, c, stal, stpl, domain_label
        # Office
        elif prob <= (self.input_ids.shape[0] + self.input_ids2.shape[0] + self.input_ids3_.shape[0]) / all_size:
            input_, mask, seg, r, c, stal, stpl = self.next_batch_office()
            for i in range(self.batch_size):
                domain_label[i, 2] = 1
            return input_, mask, seg, r, c, stal, stpl, domain_label
        # Spec
        elif prob <= (self.input_ids.shape[0] + self.input_ids2.shape[0] + self.input_ids3_.shape[0] \
                + self.input_ids4.shape[0]) / all_size:
            input_, mask, seg, r, c, stal, stpl = self.next_batch_spec()
            for i in range(self.batch_size):
                domain_label[i, 3] = 1
            return input_, mask, seg, r, c, stal, stpl, domain_label
        # Law
        else:
            input_, mask, seg, r, c, stal, stpl = self.next_batch_law()
            for i in range(self.batch_size):
                domain_label[i, 4] = 1
            return input_, mask, seg, r, c, stal, stpl, domain_label