import random

def pro_process(sentence, pro=True):
    import re
    if pro:
        sentence = re.sub(r'(，){3,}，', "。", sentence)
        sentence = re.sub(r'(噢){3,}', "噢噢噢 。", sentence)
        sentence = re.sub(r'(喵){3,}', "喵喵喵。", sentence)
        sentence = re.sub(r'(傻逼){3,}', "傻逼  。", sentence)
        sentence = re.sub(r'(？！){3,}', "？！", sentence)
        sentence = re.sub(r'(嗯){3,}', "嗯嗯嗯 。", sentence)
        sentence = re.sub(r'(啦){3,}', "啦啦啦。", sentence)
        sentence = re.sub(r'(哭着){3,}', "哭着。", sentence)
        sentence = re.sub(r'(我叫){3,}', "我叫。", sentence)
        sentence = re.sub(r'(B，){3,}', "B。", sentence)
        sentence = re.sub(r'(哈哈哈){3,}', "哈哈哈。", sentence)
        sentence = re.sub(r'(对){3,}', "对", sentence)
        sentence = re.sub(r'(宅){3,}', "宅。", sentence)
        sentence = re.sub(r'(啊啊啊){3,}', "啊啊啊啊啊啊啊啊", sentence)
        sentence = re.sub(r'(好可爱){2,}', "好可爱。", sentence)
        sentence = re.sub(r'(找){3,}', "找。", sentence)
        sentence = re.sub(r'(饿){3,}', "饿。", sentence)
        sentence = re.sub(r'(噗){3,}', "噗噗噗。", sentence)
        sentence = re.sub(r'(嘻){3,}', "嘻嘻嘻。", sentence)
        sentence = re.sub(r'(捏){3,}', "捏捏捏。", sentence)
        sentence = re.sub(r'(吹){3,}', "吹吹吹。", sentence)
        sentence = re.sub(r'(爱你){3,}', "爱你。", sentence)
        sentence = re.sub(r'(卡){3,}', "卡卡卡。", sentence)
        sentence = re.sub(r'(有爱，){3,}', "有爱。", sentence)
        sentence = re.sub(r'(听着，){3,}', "听着。", sentence)
        sentence = re.sub(r'(一张，){3,}', "一张。", sentence)
        sentence = re.sub(r'(听着，){3,}', "听着。", sentence)
        sentence = re.sub(r'(等我，){3,}', "等我。", sentence)
        sentence = re.sub(r'(各种，){3,}', "各种。", sentence)
    return sentence

def get_distinct_sample(path=None):
    def get_distinct_unigram(sentences):
        word_dict = {}
        length = 0
        for sentence in sentences:
            sentence = list(sentence)
            length += len(sentence)
            for word in sentence:
                if word not in word_dict:
                    word_dict[word] = 1
        return len(word_dict) / length

    def get_distinct_bigram(sentences):
        bigram_dict = {}
        length = 0
        for sentence in sentences:
            length += len(sentence) - 1
            for idx in range(len(sentence) - 1):
                bi_gram = sentence[idx:idx + 2]
                if bi_gram not in bigram_dict:
                    bigram_dict[bi_gram] = 1
        return len(bigram_dict) / length

    if path:
        hyb_result = []
        with open(path, "r", encoding="utf-8") as hyb_result_file:
            for idx, line in enumerate(hyb_result_file):
                line = "".join(line[:-1].split())
                line = pro_process(line, False)
                hyb_result.append(line)
            mean = get_distinct_unigram(hyb_result)
            print("uni: ", mean)
            mean = get_distinct_bigram(hyb_result)
            print("bi: ", mean)


def calculate_bleu(ref_file, trans_file, subword_option=None, max_order=1):
    import evaluation_utils

    def get_sample_test_set(before_file, after_file, sample=False):
        with open(before_file, "r", encoding="utf-8") as all_file:
            with open("./data/STC/test/test_id", "r", encoding="utf-8") as id_file:
                with open(after_file, "w", encoding="utf-8") as sample_file:
                    if sample:
                        id_list = []
                        for id in id_file:
                            id = id.strip()
                            id_list.append(int(id))
                        for idx, line in enumerate(all_file):
                            if idx in id_list:
                                line = line.strip()
                                # line = "".join(line.split())
                                line = pro_process(line, False)
                                sample_file.write(line + "\n")
                    else:
                        for idx, line in enumerate(all_file):
                            line = line.strip()
                            # line = "".join(line.split())
                            line = pro_process(line, False)
                            sample_file.write(line + "\n")

    if ref_file and trans_file:
        score = evaluation_utils.evaluate(ref_file, trans_file, subword_option=subword_option, max_order=max_order)
        # new_ref_file = ref_file + ".sampled"
        # new_trans_file = trans_file + ".sampled"
        # get_sample_test_set(ref_file, new_ref_file, sample=False)
        # get_sample_test_set(trans_file, new_trans_file, sample=False)
        # score = evaluation_utils.evaluate(new_ref_file, new_trans_file, subword_option=subword_option, max_order=max_order)
        print("%.2f" % score)

get_distinct_sample("./output/ReBoost/inference/result.beam_10.test")
calculate_bleu("./data/STC/test/cmnt.test.retrieval", "./output/ReBoost/inference/result.beam_10.test", max_order=1)
calculate_bleu("./data/STC/test/cmnt.test.retrieval", "./output/ReBoost/inference/result.beam_10.test", max_order=2)
calculate_bleu("./data/STC/test/cmnt.test.retrieval", "./output/ReBoost/inference/result.beam_10.test", max_order=3)
calculate_bleu("./data/STC/test/cmnt.test.retrieval", "./output/ReBoost/inference/result.beam_10.test", max_order=4)