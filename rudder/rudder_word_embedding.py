from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings, BertEmbeddings
import numpy as np
# initialize the word embeddings
glove_embedding = BertEmbeddings('bert-base-uncased')
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')
import torch
from sklearn.decomposition import PCA
# initialize the document embeddings, mode = mean
document_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                              flair_embedding_backward,
                                              flair_embedding_forward])
pca = PCA(128)
updated_dict = False
embDict={}
def update_dict(dic:dict):
    print('updating dict to pca')
    global embDict
    keys, values = dic.keys(), dic.values()
    full_embeddings = []
    for k in dic.keys():
        sentence = Sentence(k)
        document_embeddings.embed(sentence)
        full_embeddings.append(sentence.get_embedding().numpy())
    pca_mat = pca.fit_transform(np.array(full_embeddings))
    embDict = dict(zip(keys, torch.tensor(pca_mat,device="cuda" if torch.cuda.is_available() else "cpu")))


def transform_obs(obss):
    global embDict,updated_dict
    # embs=[document_embeddings.embed(Sentence(obs["mission"])) for obs in obss]
    embs=[]
    # pca_embs =[]
    for obs in obss:
        mission=obs["mission"]
        if mission not in embDict.keys():
            sentence=Sentence(obs["mission"])
            document_embeddings.embed(sentence)
            emb=sentence.get_embedding()[:128]
            # emb_for_pca = sentence.get_embedding().numpy()
            # pca_embs.append(emb_for_pca)
            embDict[mission]=emb
        else:
            emb=embDict[mission]
        embs.append(emb)
    # if not updated_dict and len(embDict.keys()) == 3*6*3*6:
    # # if not updated_dict and len(embDict.keys()) >= 128:
    #     embDict = update_dict(embDict)
    #     updated_dict = True
    # print("dict size ",len(embDict))
    return embs