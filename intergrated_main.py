import myelynn_DataHolder2 as DataHolder
import tensorflow as tf
import myelynn_main as QA_model
import optimization_ as optimization
import Chuncker
import modeling as modeling
import numpy as np
from utils import Fully_Connected
import tokenization
from transformers import AutoTokenizer, AutoModel
from evaluate2 import exact_match_score, f1_score
import os

def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def get_variables_with_name(name, train_only=True, printable=False):
    """Get variable list by a given name scope.
    Examples
    ---------
    >>> dense_vars = tl.layers.get_variable_with_name('dense', True, True)
    """
    print("  [*] geting variables with %s" % name)
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try:  # TF1.0
            t_vars = tf.global_variables()
        except:  # TF0.12
            t_vars = tf.all_variables()

    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)

def kl(x, y):
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y, allow_nan_stats=False)

def kl_coef(i):
    # coef for KL annealing
    # reaches 1 at i = 22000
    # https://github.com/kefirski/pytorch_RVAE/blob/master/utils/functional.py
    return (tf.tanh((i - 3500) / 1000) + 1) / 2

def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

class intergrated_model:
    def __init__(self, FirstTraining=True):
        self.firstTrainig = FirstTraining
        self.chuncker = Chuncker.Chuncker()
        self.QA = QA_model.KoNET(True)
        self.processor = DataHolder.DataHolder()

        self.save_path = './model/integrated_model_v2.ckpt'
        self.bert_path = '../pretrain_models/kobigbird/kobigbird.ckpt'

    def get_adv_output_(self, bert_config, input_tensor, domain_label, dis_lambda, global_step):
        """Get loss and log probs for the masked LM."""
        domain_num = 5
        with tf.variable_scope("cls/domain_classification"):
            dis_lambda = dis_lambda * kl_coef(global_step)
            output_weights = tf.get_variable(
                "output_weights",
                shape=[domain_num, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range))
            output_bias = tf.get_variable(
                "output_bias", shape=[domain_num], initializer=tf.zeros_initializer())

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.softmax(logits, axis=-1)
            domain_label = tf.ones_like(domain_label, dtype=tf.float32)
            domain_label = tf.reshape(domain_label, shape=[-1, domain_num])

            per_example_loss = kl(log_probs, tf.ones_like(log_probs, dtype=tf.float32))
            loss = per_example_loss * tf.cast(dis_lambda, dtype=tf.float32)
            loss = tf.reduce_mean(loss)

        return loss, per_example_loss, log_probs

    def get_get_discrimination_output_(self, bert_config, input_tensor, domain_label, dis_lambda, global_step):
        """Get loss and log probs for the masked LM."""
        domain_num = 5
        with tf.variable_scope("cls/domain_classification", reuse=True):
            dis_lambda = dis_lambda * kl_coef(global_step)
            output_weights = tf.get_variable(
                "output_weights",
                shape=[domain_num, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range))
            output_bias = tf.get_variable(
                "output_bias", shape=[domain_num], initializer=tf.zeros_initializer())

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            log_probs = tf.reshape(log_probs, shape=[self.processor.batch_size, domain_num])
            domain_label = tf.reshape(domain_label, shape=[-1, domain_num])

            per_example_loss = -tf.reduce_sum(log_probs * domain_label, axis=[-1])
            loss = tf.reduce_mean(per_example_loss)

        return loss, per_example_loss, log_probs

    def get_verify_answer(self, model_output, scope='verification_block', is_training=False):
        """Get loss and log probs for the next sentence prediction."""

        keep_prob = 0.85

        if is_training is False:
            keep_prob = 1.0

        with tf.variable_scope(scope):
            model_output = Fully_Connected(model_output, output=768, name='hidden1', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=512, name='hidden2', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=256, name='hidden3', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            log_probs = Fully_Connected(model_output, output=2, name='pointer_start1', activation=None, reuse=False)

        return log_probs

    def Training(self, is_Continue=False, training_epoch=130001):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.98
        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base_rr.json')

        with tf.Session(config=config) as sess:
            model, bert_variables_qa, sequence_output = self.QA.create_model(self.QA.input_ids, self.QA.input_mask,
                                                                             self.QA.input_segments, is_training=True,
                                                                             scope_name='bert')
            with tf.variable_scope("adapter_structure"):
                column_memory, row_memory = self.QA.Table_Memory_Network(model.get_sequence_output(),
                                                                         hops=2,
                                                                         hidden_size=768)
                row_one_hot = tf.one_hot(self.QA.input_rows, depth=100)  # input_rows: [B, S] => [B, S, 100]
                column_one_hot = tf.one_hot(self.QA.input_cols, depth=50)  # [B, S, 50]

                column_memory = tf.matmul(column_one_hot, column_memory) # [B, S, 50] X [B, 50, H] => [B, S, H]: BERT output size
                row_memory = tf.matmul(row_one_hot, row_memory)
                sequence_output = tf.concat([column_memory, row_memory, sequence_output], axis=2)

                probs_start, probs_stop = self.QA.get_qa_probs(sequence_output, scope='text_layer', is_training=True)
            # sequence_output = model.get_sequence_output()
            global_step = tf.Variable(0, name='global_step', trainable=False)
            global_step_ = tf.Variable(0, name='global_step_', trainable=False)
            global_step2 = tf.Variable(0, name='global_step2', trainable=False)

            loss_qa, _, _ = self.QA.get_qa_loss(probs_start, probs_stop)
            loss_qa = tf.reduce_mean(loss_qa)
            loss_adv, _, _ = self.get_adv_output_(bert_config, model.get_pooled_output(), self.QA.domain_label,
                                                  dis_lambda=0.5, global_step=global_step)
            loss_dis, _, _ = self.get_get_discrimination_output_(bert_config, model.get_pooled_output(),
                                                                 self.QA.domain_label,
                                                                 dis_lambda=0.5, global_step=global_step)
            loss_first = loss_qa + loss_adv
            loss_second = loss_dis
            learning_rate = 2e-5  # 0.00005

            bert_vars = get_variables_with_name('bert')
            output_vars = get_variables_with_name('adapter_structure')
            output_vars.extend(bert_vars)
            disc_vars = get_variables_with_name('cls/domain_classification')

            optimizer_qa = optimization.create_optimizer(loss=loss_first, init_lr=3e-5, num_train_steps=125000,
                                                         num_warmup_steps=5000, use_tpu=False, var_list=output_vars,
                                                         global_step=global_step_)
            optimizer_disc = optimization.create_optimizer(loss=loss_second, init_lr=3e-5, num_train_steps=125000,
                                                           num_warmup_steps=5000, use_tpu=False, var_list=disc_vars,
                                                           global_step=global_step2)
            sess.run(tf.initialize_all_variables())

            if self.firstTrainig is True:
                bert_variables = get_variables_with_name(name='bert')
                saver = tf.train.Saver(bert_variables)
                saver.restore(sess, self.bert_path)
                print('BERT restored')

            if is_Continue is True:
                saver = tf.train.Saver()
                saver.restore(sess, self.save_path)

            for i in range(training_epoch):
                # 라벨 정보 바뀌어야함
                input_ids, input_masks, input_segments, input_rows, input_cols, start_label, stop_label, domain_label \
                    = self.processor.next_batch_all_adv2()

                feed_dict = {self.QA.input_ids: input_ids,
                             self.QA.input_segments: input_segments,
                             self.QA.start_label: start_label, self.QA.stop_label: stop_label,
                             # self.input_weights: input_weights,
                             self.QA.input_rows: input_rows, self.QA.input_cols: input_cols,
                             # self.rank_label: vf_label
                             self.QA.domain_label: domain_label
                             }

                # try:
                g1, loss_qa_, _ = sess.run([global_step, loss_first, optimizer_qa], feed_dict=feed_dict)
                g2, loss_disc_, _ = sess.run([global_step, loss_second, optimizer_disc], feed_dict=feed_dict)
                # except:
                #     g1 = 0
                #     g2 = 0
                #     loss_qa_ = 0
                #     loss_disc_ = 0

                # print(pred[0, 0:10])
                # print(pred2[0, 0:10])
                # print(column_label[0, 0:10])
                print(g1, i)
                print(g2, loss_qa_, loss_disc_)

                # print('-------')

                # if i % 1000 == 0 and i > 100:
                #     print('saved!')
                #     saver = tf.train.Saver()
                #     saver.save(sess, self.save_path)
                if i % 10000 == 0 and i != 0:
                    print('saved!')
                    saver = tf.train.Saver()
                    saver.save(sess, self.save_path)

    def eval_with_span_test(self):
        dataholder = DataHolder.DataHolder()

        vocab = tokenization.load_vocab(vocab_file='vocab_bigbird.txt')
        tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.98
        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base_rr.json')

        tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
        tokenizer.add_tokens('[table]')
        tokenizer.add_tokens('[/table]')
        tokenizer.add_tokens('[list]')
        tokenizer.add_tokens('[/list]')
        tokenizer.add_tokens('[h3]')
        tokenizer.add_tokens('[td]')

        em_total = 0
        f1_total = 0
        epo = 0

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.QA.create_model(self.QA.input_ids, self.QA.input_mask,
                                                                          self.QA.input_segments, is_training=False)
            with tf.variable_scope("adapter_structure"):
                input_shape = modeling.get_shape_list(sequence_output, expected_rank=3)
                batch_size = input_shape[0]
                seq_length = input_shape[1]

                column_memory, row_memory = self.QA.Table_Memory_Network(model.get_sequence_output(), hops=2,
                                                                         hidden_size=768)

                row_one_hot = tf.one_hot(self.QA.input_rows, depth=100)
                column_one_hot = tf.one_hot(self.QA.input_cols, depth=50)

                column_memory = tf.matmul(column_one_hot, column_memory)
                row_memory = tf.matmul(row_one_hot, row_memory)

                sequence_output = tf.concat([column_memory, row_memory, sequence_output], axis=2)
                # sequence_output = Fully_Connected(sequence_output, output=768, name='column_wise', activation=None)

                # sequence_output = tf.concat([names_embeddings, sequence_output], axis=2)
                prob_start, prob_stop = self.QA.get_qa_probs(sequence_output, scope='text_layer', is_training=False)

                prob_start = tf.nn.softmax(prob_start, axis=-1)
                prob_stop = tf.nn.softmax(prob_stop, axis=-1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            # 1. Law test
            for _ in range(300):
                input_ids, input_segments, input_rows, input_cols, answer_text = dataholder.test_batch_law()

                feed_dict = {self.QA.input_ids: input_ids, self.QA.input_segments: input_segments,
                             self.QA.input_rows: input_rows, self.QA.input_cols: input_cols,

                             }

                question_end = 0

                if input_ids.shape[0] > 0:
                    probs_start, probs_stop = sess.run([prob_start, prob_stop], feed_dict=feed_dict)

                    probs_start = np.array(probs_start, dtype=np.float32)
                    probs_stop = np.array(probs_stop, dtype=np.float32)

                    for j in range(input_ids.shape[0]):
                        for k in range(1, input_ids.shape[1]):
                            probs_start[j, k] = 0
                            probs_stop[j, k] = 0
                            question_end += 1

                            if tokenizer.decode(input_ids[j, k]) == '[SEP]':
                                break

                    # self.chuncker.get_feautre(answer_text)

                    prob_scores = []
                    c_scores = []

                    for j in range(input_ids.shape[0]):
                        # paragraph ranking을 위한 score 산정기준
                        # score2 = ev_values[j, 0]
                        score2 = 2 - (probs_start[j, 0] + probs_stop[j, 0])

                        prob_scores.append(score2)
                        # c_scores.append(self.chuncker.get_chunk_score(sequences[j]))

                    if True:
                        for j in range(input_ids.shape[0]):
                            probs_start[j, 0] = -999
                            probs_stop[j, 0] = -999

                        # CLS 선택 무효화
                        prediction_start = probs_start.argmax(axis=1)
                        prediction_stop = probs_stop.argmax(axis=1)

                        answers = []
                        scores = []
                        candi_scores = []

                        for j in range(input_ids.shape[0]):
                            answer_start_idx = prediction_start[j]
                            answer_stop_idx = prediction_stop[j]

                            if input_cols[0, answer_start_idx] != input_cols[0, answer_stop_idx]:
                                answer_stop_idx2 = answer_stop_idx
                                answer_stop_idx = answer_start_idx
                                answer_start_idx2 = answer_stop_idx2

                                for k in range(answer_start_idx + 1, input_ids.shape[1]):
                                    if input_cols[0, k] == input_cols[0, answer_start_idx]:
                                        answer_stop_idx = k
                                    else:
                                        break

                                for k in reversed(list(range(0, answer_stop_idx2 - 1))):
                                    if input_cols[0, k] == input_cols[0, answer_stop_idx2]:
                                        answer_start_idx2 = k
                                    else:
                                        break

                                prob_1 = probs_start[0, answer_start_idx] + probs_stop[0, answer_stop_idx]

                                prob_2 = probs_start[0, answer_start_idx2] + probs_stop[0, answer_stop_idx2]

                                if prob_2 > prob_1:
                                    answer_start_idx = answer_start_idx2
                                    answer_stop_idx = answer_stop_idx2

                            score = probs_start[j, answer_start_idx]
                            scores.append(score * 1)
                            candi_scores.append(score * 1)

                            if answer_start_idx > answer_stop_idx:
                                answer_stop_idx = answer_start_idx + 15
                            # if f_tokenizer.inv_vocab[input_ids[j, answer_start_idx]] == '[p]':
                            #     for k in range(answer_start_idx, input_ids.shape[1]):
                            #         # if f_tokenizer.inv_vocab[input_ids[j, k]] == '[/p]':
                            #             answer_stop_idx = k
                            #             break

                            answer = ''

                            if answer_stop_idx + 1 >= input_ids.shape[1]:
                                answer_stop_idx = input_ids.shape[1] - 2

                            for k in range(answer_start_idx, answer_stop_idx + 1):
                                # tok = f_tokenizer.inv_vocab[input_ids[j, k]]
                                tok = tokenizer.decode(input_ids[j, k])
                                if len(tok) > 0:
                                    if tok[0] != '#':
                                        answer += ' '
                                answer += str(tok).replace('##', '')

                            answers.append(answer)

                    if len(answers) > 0:
                        answer_candidates = []
                        candidates_scores = []

                        for _ in range(1):
                            m_s = -99
                            m_ix = 0

                            for q in range(len(scores)):
                                if m_s < scores[q]:
                                    m_s = scores[q]
                                    m_ix = q

                            answer_candidates.append(answers[m_ix])
                            candidates_scores.append(candi_scores[m_ix])

                        a1 = [0]
                        a2 = [0]
                        answer_text = str(answer_text[0])

                        for a_c in answer_candidates:
                            if a_c.find('<table>') != -1:
                                continue
                            a1.append(exact_match_score(prediction=a_c, ground_truth=answer_text))
                            a2.append(f1_score(prediction=a_c, ground_truth=answer_text))

                        em_total += max(a1)
                        if max(a1) > 0.99: a2.append(1.0)
                        f1_total += max(a2)
                        epo += 1

                        for j in range(input_ids.shape[0]):
                            check = 'None'
                            answer_text = answer_text.replace('<a>', '').replace('</a>', '')

                            f1_ = f1_score(prediction=answers[j], ground_truth=answer_text)

                            text = answers[j]

                        question_text = ''
                        for qu in range(1, question_end):
                            tmp = tokenizer.decode(input_ids[0][qu])
                            if tmp[0] != '#': question_text += ' '
                            question_text += str(tmp).replace('##', '')

                        print('-----\n', epo)
                        print(f'score: {scores[j]}\nThis F1: {f1_}')
                        print('question:', question_text)
                        print('answer:', answer_text)
                        print('predict: ', answer_candidates[0])
                        print('EM:', em_total / epo)
                        print('F1:', f1_total / epo)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
int_model = intergrated_model(True)
int_model.Training(False, 111000)
# int_model.eval_with_span_test()
