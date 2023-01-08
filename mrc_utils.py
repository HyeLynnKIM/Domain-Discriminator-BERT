import numpy as np
from HTML_Utils import answer_re_touch


def propagate_answer(probs_start, probs_stop, probs_vf, input_ids, rank_scores, chunk_scores, f_tokenizer):
    for j in range(input_ids.shape[0]):
        for k in range(1, input_ids.shape[1]):
            probs_start[j, k] = 0
            probs_stop[j, k] = 0

            if input_ids[j, k] == 3:
                break

    prob_scores = []
    for j in range(input_ids.shape[0]):
        # paragraph ranking을 위한 score 산정기준
        score2 = -(probs_start[j, 0] + probs_stop[j, 0])
        prob_scores.append(score2)

    for j in range(input_ids.shape[0]):
        probs_start[j, 0] = -999
        probs_stop[j, 0] = -999

    # CLS 선택 무효화

    prediction_start = probs_start.argsort(axis=1)[:, 512 - 5:512]
    prediction_stop = probs_stop.argsort(axis=1)[:, 512 - 5:-512]

    answers = []
    scores = []
    candi_scores = []

    for j in range(input_ids.shape[0]):
        answer_start_idxs = prediction_start[j]
        answer_end_idxs = prediction_stop[j]

        probs = []
        idxs = []

        for start_idx in answer_start_idxs:
            for end_idx in answer_end_idxs:
                if start_idx > end_idx:
                    continue
                if end_idx - start_idx > 30:
                    continue

                idxs.append([start_idx, end_idx])
                probs.append(probs_start[j, start_idx] + probs_stop[j, end_idx])

        if len(probs) == 0:
            answer_start_idx = probs_start.argmax(axis=1)[j]
            answer_stop_idx = probs_stop.argmax(axis=1)[j]
        else:
            idx = np.array(probs).argmax()
            answer_start_idx = idxs[idx][0]
            answer_stop_idx = idxs[idx][1]

        score = chunk_scores[j] * probs_vf[j] * probs_start[j, answer_start_idx] * \
                probs_stop[j, answer_stop_idx] * rank_scores[j]

        scores.append(score)
        candi_scores.append(score)

        answer = ''
        if answer_stop_idx + 1 >= input_ids.shape[1]:
            answer_stop_idx = input_ids.shape[1] - 2

        for k in range(answer_start_idx, answer_stop_idx + 1):
            tok = f_tokenizer.inv_vocab[input_ids[j, k]]
            if len(tok) > 0:
                if tok[0] != '#':
                    answer += ' '
            answer += str(f_tokenizer.inv_vocab[input_ids[j, k]]).replace('##', '')

        answers.append(answer)

    m_s = -99
    m_ix = 0

    for q in range(len(scores)):
        if m_s < scores[q]:
            m_s = scores[q]
            m_ix = q

    selected_answer = answer_re_touch(answers[m_ix])
    selected_idx = m_ix

    return selected_answer, selected_idx