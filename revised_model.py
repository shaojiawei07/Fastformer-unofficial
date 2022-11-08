# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# The following code is the Fastformer, which is build upon NRMS.


import tensorflow.keras as keras
from tensorflow.keras import layers


#from recommenders.models.newsrec.models.base_model import BaseModel
from revised_based_model import BaseModel
from recommenders.models.newsrec.models.layers import AttLayer2
from fastformer import Fastformer

__all__ = ["my_model"]



class my_model(BaseModel):
    """
    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(
        self,
        hparams,
        iterator_creator,
        seed=None,
    ):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.
        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            iterator_creator_train (object): NRMS data loader class for train data.
            iterator_creator_test (object): NRMS data loader class for test and validation data
        """
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)

        super().__init__(
            hparams,
            iterator_creator,
            seed=seed,
        )

    def _get_input_label_from_iter(self, batch_data):
        """get input and labels for trainning from iterator
        Args:
            batch data: input batch data from iterator
        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        """get input of user encoder
        Args:
            batch_data: input batch data from user iterator
        Returns:
            numpy.ndarray: input user feature (clicked title batch)
        """
        return batch_data["clicked_title_batch"]

    def _get_news_feature_from_iter(self, batch_data):
        """get input of news encoder
        Args:
            batch_data: input batch data from news iterator
        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        return batch_data["candidate_title_batch"]

    def _build_graph(self):
        """Build NRMS model and scorer.
        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.
        Args:
            titleencoder (object): the news encoder of NRMS.
        Return:
            object: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )

        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title)

        y = Fastformer(hparams.head_num, hparams.head_dim)(
            [click_title_presents] * 3
        )

        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        print("_build_userencoder","user_present.shape",user_present.shape)

        model = keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NRMS.
        Args:
            embedding_layer (object): a word embedding layer.
        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")

        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        #y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        #pred_title = Fastformer(hparams.head_num, hparams.head_dim)([y, y, y])
        y = Fastformer(hparams.head_num, hparams.head_dim)([y, y, y])

        #print("_build_newsencoder","SelfAttention output.shape",y.shape)
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        print("_build_newsencoder","pred_title.shape",pred_title.shape)

        model = keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.
        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_title_one = keras.Input(
            shape=(
                1,
                hparams.title_size,
            ),
            dtype="int32",
        )
        pred_title_one_reshape = layers.Reshape((hparams.title_size,))(
            pred_input_title_one
        )

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_newsencoder(embedding_layer) # use Fastformer
        self.userencoder = self._build_userencoder(titleencoder) # use Fastformer
        self.newsencoder = titleencoder

        user_present = self.userencoder(his_input_title)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title) # Training
        news_present_one = self.newsencoder(pred_title_one_reshape) # Inference

        preds = layers.Dot(axes=-1)([news_present, user_present]) # Training
        preds = layers.Activation(activation="softmax")(preds) # Training 

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present]) # Inference
        pred_one = layers.Activation(activation="sigmoid")(pred_one) # Inference

        model = keras.Model([his_input_title, pred_input_title], preds) # Training
        scorer = keras.Model([his_input_title, pred_input_title_one], pred_one) # Inference

        return model, scorer