import re
import collections
import math

def evaluate(ref_file, trans_file, subword_option=None, max_order=4):
    evaluation_score = _bleu(ref_file, trans_file, subword_option=subword_option, max_order=max_order)
    return evaluation_score

def _clean(sentence, subword_option):
    sentence = sentence.strip()
    # BPE
    if subword_option == "bpe":
        sentence = re.sub("@@ ", "", sentence)
    # SPM
    elif subword_option == "spm":
        sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

    return sentence

def _bleu(ref_file, trans_file, subword_option=None, max_order=4):
    """Compute BLEU scores and handling BPE."""
    max_order = max_order
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with open(reference_filename, "r", encoding="utf-8") as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, subword_option)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    with open(trans_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = _clean(line, subword_option=None)
            translations.append(line.split(" "))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = compute_bleu(per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score

def _get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)
