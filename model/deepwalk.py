import numpy as np
from gensim.models import Word2Vec

class DeepWalk:

 def __init__(self, graph, walk_numbers=10, walk_deep=20, embeding_size=128, window_size=5, workers=3, iters=5, min_count=0, sg=1, hs=1, negative=1, seed=128, compute_loss=False, is_save_sentence=False, save_sentence_path=None, **kwargs):
  """
   DeepWalk Model 函数类初始化
  :param graph: 创建的 networdx 对象
  :param walk_numbers: 每个节点创建多少个序列
  :param walk_deep: 遍历的深度
  :param embeding_size: 生成的embedding 大小「word2vec参数」
  :param window_size: 窗口大小「word2vec参数」
  :param workers: 采用多少个线程生成embedding 「word2vec参数」
  :param iters: 迭代次数「word2vec参数」
  :param min_count: 过滤词频低于改值的item「word2vec参数」
  :param sg: 1 表示 Skip-gram 0 表示 CBOW「word2vec参数」
  :param hs: 1 表示 hierarchical softmax  0 且 negative 参数不为0 的话 negative sampling 会被启用「word2vec参数」
  :param negative: 0 表示不采用，1 表示采用，建议值在 5-20 表示噪音词的个数「word2vec参数」
  :param seed: 随机初始化的种子「word2vec参数」
  :param compute_loss: 是否计算loss「word2vec参数」
  :param is_save_sentence: 是否保存序列数据
  :param save_sentence_path: 保存序列数据的路径
  :param kwargs:
  """
  self.walk_numbers = walk_numbers
  self.walk_deep = walk_deep

  self.embedding_size = embeding_size
  self.window_size = window_size
  self.workers = workers
  self.iter = iters
  self.min_count = min_count
  self.sg = sg
  self.hs = hs
  self.negative = negative
  self.seed = seed
  self.compute_loss = compute_loss

  self.graph = graph
  self.w2v_model = None
  self.is_save_sentence = is_save_sentence
  self.save_sentence_path = save_sentence_path
  self.sentences = self.gen_sentences()

 def gen_sentences(self):
  sentences = list()
  i = 0
  for node in self.graph.nodes:
   i += 1
   print("开始从节点 i={}: {} 开始生成随机游走序列！".format(i, node))
   _corpus = list()
   for walk_number in range(self.walk_numbers):
    sentence = [node]
    current_node = node
    deep = 0
    while deep < self.walk_deep:
     deep += 1
     node_nbr_list = list()
     node_weight_list = list()
     if self.graph[current_node].items().__len__() == 0:
      continue
     for nbr, weight_dict in self.graph[current_node].items():
      node_nbr_list.append(nbr)
      node_weight_list.append(weight_dict["weight"])
     node_weight_norm_list = [float(_weight) / sum(node_weight_list) for _weight in node_weight_list]
     new_current_node = np.random.choice(node_nbr_list, p=node_weight_norm_list)
     sentence.append(new_current_node)
     current_node = new_current_node
   sentences.append(sentence)
  if self.is_save_sentence:
   fw = open(self.save_sentence_path, "w")
   for sentence in sentences:
    fw.write(",".join(sentence) + "\n")
   fw.close()
  return sentences

 def train(self, **kwargs):
  kwargs["sentences"] = self.sentences
  kwargs["iter"] = self.iter
  kwargs["size"] = self.embedding_size
  kwargs["window"] = self.window_size
  kwargs["min_count"] = self.min_count
  kwargs["workers"] = self.workers
  kwargs["sg"] = self.sg
  kwargs["hs"] = self.hs
  kwargs["negative"] = self.negative
  kwargs["seed"] = self.seed
  kwargs["compute_loss"] = self.compute_loss
  model = Word2Vec(**kwargs)
  print("DeepWalk Embedding Done!")

  self.w2v_model = model
  return model

 def embedding(self, word):
  return self.w2v_model.wv[word]

 def embeddings(self):
  embedding_dict = dict()
  for node in self.graph.nodes:
   embedding_dict[node] = self.embedding(node)
  return embedding_dict

 def similarity(self, word1, word2):
  return self.w2v_model.wv.similarity(word1, word2)

 def most_similar(self, word, topn=200):
  return self.w2v_model.wv.most_similar(word, topn=topn)

 def save_embedding(self, path):
  fw = open(path, "w")
  for node in self.graph.nodes:
   fw.write(node + "\t" + ",".join(map(str, list(self.embedding(node)))) + "\n")
  fw.close()
  print("embedding save to: {} done!".format(path))

 def save_node_sim_nodes(self, path, topn=200):
  fw = open(path, "w")
  for node in self.graph.nodes:
   _list = list()
   for sim_node in self.most_similar(node, topn=topn):
    _list.append(sim_node[0] + ":" + format(sim_node[1], ".4f"))
   fw.write(node + "\t" + ",".join(_list) + "\n")
  fw.close()
  print("nodes sim nodes save to: {} done!".format(path))