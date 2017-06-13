import numpy as np

# define an interface to word definitions
#     post(a matrix U_embed, specify which dict/s): 
#          len==0 means all of them
#         # to be embedded, or further embedded
#         xxxxxxx return the finished matrix U_embed 
#           avoid embedding itself into the matrix
#           mat is modified, not copied. seriously.!!!
#     get(specify name of the dictionary, only one at a time): 
#         definition in form of list of strings
#         of entire 1k matrix
#             as a dictionary indexed by words

def embedMat(mat,*args):
    from components.w2v import i2w,w2i
    dicts=['cambridge','collins','dictionary','mw','oxford','vocabulary']
    for i,w in enumerate(i2w):
        if not len(args)==0:
            for j,dname in enumerate(args):
                dloc='crawlen/dicts/'+w+'/'+dname+'.txt'
                vec=mat[i]
                f=open(dloc,'rb')
                thedef=' '.join([s.decode().strip() for s in f.readlines()]).split(' ')
                for k,word in enumerate(thedef):
                    word=word.lower()
                    if word in w2i and word != w:
                        vec[w2i[word]]+=1
                f.close()
                # break
            # break

