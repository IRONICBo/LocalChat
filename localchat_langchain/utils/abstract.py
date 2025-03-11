import numpy as np
import networkx as nx
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from textrank4zh.util import get_similarity, AttrDict
import textrank4zh

tr4s = TextRank4Sentence()


def _patched_sort_sentences(
    sentences,
    words,
    sim_func=get_similarity,
    pagerank_config={
        "alpha": 0.85,
    },
):
    sorted_sentences = []
    _source = words
    sentences_num = len(_source)
    graph = np.zeros((sentences_num, sentences_num))

    for x in range(sentences_num):
        for y in range(x, sentences_num):
            similarity = sim_func(_source[x], _source[y])
            graph[x, y] = similarity
            graph[y, x] = similarity

    # nx_graph = nx.from_numpy_matrix(graph)
    # Patched function here
    nx_graph = nx.from_numpy_array(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)  # this is a dict
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    for index, score in sorted_scores:
        item = AttrDict(index=index, sentence=sentences[index], weight=score)
        sorted_sentences.append(item)

    return sorted_sentences


# Monkey patch to fix the bug
textrank4zh.util.sort_sentences = _patched_sort_sentences


def get_abstract(content):
    try:
        tr4s.analyze(text=content, lower=True, source="all_filters")
        # Get top 1 sentence
        abstracts = tr4s.get_key_sentences(num=3)
        print(f"Abstracts of file: {abstracts}")
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        abstracts = []

    if abstracts is None or len(abstracts) <= 0:
        return content[:10] if len(content) > 10 else content

    return abstracts[0].sentence
