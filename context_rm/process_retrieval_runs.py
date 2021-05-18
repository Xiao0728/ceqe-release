import operator


def get_topk_ret_doc_p_query(ret_result, topk):
    sorted_run = {}
    for q in ret_result:
        sorted_docs = sorted(ret_result[q].items(), key=operator.itemgetter(1), reverse=True)
        sorted_run[q] = sorted_docs[:topk]
    return sorted_run
