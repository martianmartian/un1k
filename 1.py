import os,sys
sys.path.append(os.getcwd())
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)
# np.set_printoptions(threshold=10)
np.random.seed(0)


# import index2word, word2index, empty matrix for embedding, 1k identity matrix
# from components.w2v import i2w,w2i,U_embed,W_identity


# # ===============================================
# # using the def1k interface to embed the U_embed
# # and specify which dicts needed
# import components.def1k as def1k
# def1k.embedMat(U_embed,'cambridge','collins','dictionary','mw','oxford','vocabulary')  
# np.save('saves/U_embed.npy',U_embed)
# # print(U_embed[0])
# # -----------------------------------------------



# # ===============================================
# U_embed=np.load('saves/U_embed.npy')
# vec = U_embed[w2i['gun']]
# # meaning = [i2w[i] if vec[i]!=0 else 0 for i in range(1000)]
# meaning = [i2w[i] if vec[i]>3 and vec[i]<40 else 0 for i in range(1000)]
# print(vec)
# print(meaning)
# # # -----------------------------------------------




# =========== network testing =====================
# =================================================

# # ===============================================
# from nets.rnn1_.trainer import trainer
# from nets.rnn1 import rnn
# rnnChecker=rnn(lr=0.0001)
# trainer(rnnChecker,
#     save='saves/rnnChecker/',
#     maploc='nets/rnn1_/heatmaps/',
#     test=2,cycles=2,decay=0.95,limit=1,reg=0)
# # # -----------------------------------------------

# # ===============================================
# from nets.rnn2_.trainer import trainer
# from nets.rnn2 import rnn
# crnn=rnn(lr=0.0001)
# trainer(crnn,test=2,cycles=2,decay=0.9,limit=1,reg=0)
# # # -----------------------------------------------


# ===============================================
from nets.rnn3_.trainer import trainer
from nets.rnn3 import rnn
crnn=rnn(lr=0.0001)
trainer(crnn,test=2,cycles=2,decay=0.9)
# # -----------------------------------------------


