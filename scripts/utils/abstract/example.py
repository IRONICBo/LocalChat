import sys

import numpy as np
import networkx as nx
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from textrank4zh.util import get_similarity, AttrDict
import textrank4zh

def patched_sort_sentences(sentences, words, sim_func = get_similarity, pagerank_config = {'alpha': 0.85,}):
    sorted_sentences = []
    _source = words
    sentences_num = len(_source)
    graph = np.zeros((sentences_num, sentences_num))

    for x in range(sentences_num):
        for y in range(x, sentences_num):
            similarity = sim_func( _source[x], _source[y] )
            graph[x, y] = similarity
            graph[y, x] = similarity

    # nx_graph = nx.from_numpy_matrix(graph)
    nx_graph = nx.from_numpy_array(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)              # this is a dict
    sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)

    for index, score in sorted_scores:
        item = AttrDict(index=index, sentence=sentences[index], weight=score)
        sorted_sentences.append(item)

    return sorted_sentences

textrank4zh.util.sort_sentences = patched_sort_sentences

text = """
从DeepSeek-3FS聊到AI存储

目录

收起

横空出世: 3FS

AI 需要什么样的存储？

AI训练框架 + 存储结合

AI训练阶段及存储需求

AI与CephFS

CephFS应用在AI优点

CephFS缺点

CephFS可能的优化

典型AI存储系统

3FS 现状

参考文献

横空出世: 3FS

近两个月，DeepSeek的热度值爆表。尤其令我们存储人欣喜的是，DeepSeek竟然把一向养在深闺无人识的分布式存储推到了台前。

3FS的github关注度之大，在开源分布式存储项目算是前无古人了，估计也很难有来者。开源3天的star数就超过了一众已经耕耘数年的开源存储项目。

朋友圈和各个技术交流群里也到处都是谈论3FS的声音。所以也来蹭蹭热度，不过我主要不是想讨论3FS，而是想讨论讨论大模型背后的存储以及本人实际用的比较多的存储系统在AI这块的应用。毕竟讨论3FS的技术文章已经很多了，珠玉在前，就不献丑了。

DeepSeek 3FS：端到端无缓存的存储新范式

"""

tr4s = TextRank4Sentence()
tr4s.analyze(text=text, lower=True, source = 'all_filters')
for item in tr4s.get_key_sentences(num=3):
    print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重