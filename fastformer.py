
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import *
from keras.models import Model, load_model
from keras import backend as K
from sklearn.metrics import *
from keras.optimizers import *


class Fastformer(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        self.now_input_shape=None

        print("Use Fastformer")

        super(Fastformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.now_input_shape=input_shape
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True) 
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True) 
        self.Wq = self.add_weight(name='Wq', 
                                  shape=(self.output_dim,self.nb_head),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wk = self.add_weight(name='Wk', 
                                  shape=(self.output_dim,self.nb_head),
                                  initializer='glorot_uniform',
                                  trainable=True)
        
        self.WP = self.add_weight(name='WP', 
                                  shape=(self.output_dim,self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        
        
        super(Fastformer, self).build(input_shape)
        
    def call(self, x):
    	# where is the value transformation?
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
        elif len(x) == 2:
            Q_seq, K_seq = x
        else:
        	assert len(x) == 3
        # elif len(x) == 4:
        #     Q_seq,K_seq,Q_mask,K_mask = x #different mask lengths, reserved for cross attention

        Q_seq = K.dot(Q_seq, self.WQ)  
        V_seq = K.dot(V_seq, self.WV) 

        Q_seq_reshape = K.reshape(Q_seq, (-1, self.now_input_shape[0][1], self.nb_head*self.size_per_head))

        Q_att=  K.permute_dimensions(K.dot(Q_seq_reshape, self.Wq),(0,2,1))/ self.size_per_head**0.5

        if len(x)  == 4:
            Q_att = Q_att-(1-K.expand_dims(Q_mask,axis=1))*1e8

        Q_att = K.softmax(Q_att)
        Q_seq = K.reshape(Q_seq, (-1,self.now_input_shape[0][1], self.nb_head, self.size_per_head))
        V_seq = K.reshape(V_seq, (-1,self.now_input_shape[0][1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1,self.now_input_shape[1][1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))

        Q_att = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=3),self.size_per_head,axis=3))(Q_att)
        global_q = K.sum(multiply([Q_att, Q_seq]),axis=2)
        
        global_q_repeat = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=2), self.now_input_shape[1][1],axis=2))(global_q)

        QK_interaction = multiply([K_seq, global_q_repeat])
        QK_interaction_reshape = K.reshape(QK_interaction, (-1, self.now_input_shape[0][1], self.nb_head*self.size_per_head))
        K_att = K.permute_dimensions(K.dot(QK_interaction_reshape, self.Wk),(0,2,1))/ self.size_per_head**0.5
        
        if len(x)  == 4:
            K_att = K_att-(1-K.expand_dims(K_mask,axis=1))*1e8
            
        K_att = K.softmax(K_att)

        K_att = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=3),self.size_per_head,axis=3))(K_att)

        global_k = K.sum(multiply([K_att, QK_interaction]),axis=2)
     
        global_k_repeat = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=2), self.now_input_shape[0][1],axis=2))(global_k)
        #Q!=V
        

        QKQ_interaction = multiply([global_k_repeat, V_seq]) # Not Q_seq
        #QKQ_interaction = multiply([global_k_repeat, Q_seq]) # Not Q_seq
        QKQ_interaction = K.permute_dimensions(QKQ_interaction, (0,2,1,3))
        QKQ_interaction = K.reshape(QKQ_interaction, (-1,self.now_input_shape[0][1], self.nb_head*self.size_per_head))
        QKQ_interaction = K.dot(QKQ_interaction, self.WP)
        QKQ_interaction = K.reshape(QKQ_interaction, (-1,self.now_input_shape[0][1], self.nb_head,self.size_per_head))
        QKQ_interaction = K.permute_dimensions(QKQ_interaction, (0,2,1,3))
        QKQ_interaction = QKQ_interaction+Q_seq
        QKQ_interaction = K.permute_dimensions(QKQ_interaction, (0,2,1,3))
        QKQ_interaction = K.reshape(QKQ_interaction, (-1,self.now_input_shape[0][1], self.nb_head*self.size_per_head))

        #many operations can be optimized if higher versions are used. 
        
        return QKQ_interaction
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


if __name__ == "__main__":
	model = Fastformer(nb_head = 20, size_per_head = 20)