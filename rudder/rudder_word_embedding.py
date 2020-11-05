from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings, BertEmbeddings

# initialize the word embeddings
glove_embedding = BertEmbeddings('bert-base-uncased')
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')

# initialize the document embeddings, mode = mean
document_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                              flair_embedding_backward,
                                              flair_embedding_forward])
embDict={}
def transform_obs(obss):
    # embs=[document_embeddings.embed(Sentence(obs["mission"])) for obs in obss]
    embs=[]
    for obs in obss:
        mission=obs["mission"]
        if mission not in embDict.keys():
            sentence=Sentence(obs["mission"])
            document_embeddings.embed(sentence)
            emb=sentence.get_embedding()[:128]
            embDict[mission]=emb
        else:
            emb=embDict[mission]
        embs.append(emb)
    return embs