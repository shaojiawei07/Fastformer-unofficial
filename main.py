import sys
import os
import numpy as np
import zipfile
from tqdm import tqdm
import scrapbook as sb
#from data_loader import
import tensorflow as tf
import datetime
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
#from recommenders.models.newsrec.models.nrms import NRMSModel
from revised_model import my_model
#from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from revised_mind_iterator import MINDIterator
#from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

epochs = 5
seed = 40
batch_size = 64

# Options: demo, small, large
MIND_type = 'large'
data_path = './dataset_large'
model_path = f'./logs/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + "_epoch_" +str(epochs) +"_batch_"+str(batch_size)
os.makedirs(model_path, exist_ok=True)

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')

wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')

# mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)
# if not os.path.exists(train_news_file):
#     download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
# if not os.path.exists(valid_news_file):
#     download_deeprec_resources(mind_url, \
#                                os.path.join(data_path, 'valid'), mind_dev_dataset)
# if not os.path.exists(yaml_file):
#     download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
#                                os.path.join(data_path, 'utils'), mind_utils)

print('\n\n\n')


#Init

hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
print(hparams,"hparams \n")


# train the NRMS model 
iterator = MINDIterator
model = my_model(hparams, iterator, seed=seed)
#print(model.run_eval(valid_news_file, valid_behaviors_file))

#%%time
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)



# save the model
model.model.save_weights(os.path.join(model_path, "ckpt"))

# remove the news from the val set
# test on the test set
print("test")
test_data_path = './dataset_large'
test_news_file = os.path.join(test_data_path, 'test', r'news.tsv')
test_behaviors_file = os.path.join(test_data_path, 'test', r'behaviors.tsv')
#model.test_iterator.init_news(test_news_file)
#model.test_iterator.init_behaviors(test_behaviors_file)
group_impr_indexes, group_labels, group_preds = model.run_fast_eval("test" ,test_news_file, test_behaviors_file)

# write the results to the prediction.txt
with open(os.path.join(model_path, 'prediction.txt'), 'w') as f:
    for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
        impr_index += 1
        pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
        pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
        f.write(' '.join([str(impr_index), pred_rank])+ '\n')

f = zipfile.ZipFile(os.path.join(model_path, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
f.write(os.path.join(model_path, 'prediction.txt'), arcname='prediction.txt')
f.close()


