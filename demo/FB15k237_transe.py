# coding=utf-8
import os

from tf_kge.model.transe import TransE
from google.colab import files
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tf_kge

from tqdm import tqdm
import tensorflow as tf
from tf_kge.dataset.FB15k237 import FB15k237Dataset
from tf_kge.utils.sampling_utils import entity_negative_sampling
import numpy as np

train_kg, test_kg, valid_kg, entity_indexer, relation_indexer = FB15k237Dataset().load_data()

embedding_size = 50
margin = 1.0
train_batch_size = 8000
test_batch_size = 100
best_mrr_module = 0
best_mr_module = 0
best_module_hits_ten = 0
best_module_hits_three = 0
best_module_hits_one = 0
best_module_epoch = 0

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)


def compute_distance(a, b):
    # return tf.reduce_sum((a - b) ** 2, axis=-1)
    return tf.reduce_sum(tf.abs(a - b), axis=-1)


model = TransE(train_kg.num_entities, train_kg.num_relations, embedding_size)

for epoch in range(10000):

    for step, (batch_h, batch_r, batch_t) in enumerate(
            tf.data.Dataset.from_tensor_slices((train_kg.h, train_kg.r, train_kg.t)).shuffle(10000).batch(
                    train_batch_size)):
        target_entity_type = "head" if np.random.randint(0, 2) == 0 else "tail"
        # for target_entity_type in ["head", "tail"]:
        with tf.GradientTape() as tape:
            if target_entity_type == "tail":
                batch_source = batch_h
                batch_target = batch_t
            else:
                batch_source = batch_t
                batch_target = batch_h
            
            
            batch_neg_target = entity_negative_sampling(batch_source, batch_r, kg=train_kg,
                                                        target_entity_type=target_entity_type, filtered=True)

            translated = model([batch_source, batch_r], target_entity_type=target_entity_type)
            embedded_target = model.embed_norm_entities(batch_target)
            embedded_neg_target = model.embed_norm_entities(batch_neg_target)

            pos_dis = compute_distance(translated, embedded_target)
            neg_dis = compute_distance(translated, embedded_neg_target)

            losses = tf.maximum(margin + pos_dis - neg_dis, 0.0)
            loss = tf.reduce_mean(losses)

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 200 == 0:
            print("epoch = {}\tstep = {}\tloss = {}".format(epoch, step, loss))

    if epoch % 200 == 0:
        
        hits_at_ten, hits_at_three, hits_at_one = [], [], []
        mean_ranks = []
        mrr = []
        normed_entity_embeddings = tf.math.l2_normalize(model.entity_embeddings, axis=-1)

        for target_entity_type in ["head", "tail"]:
            for test_step, (batch_h, batch_r, batch_t) in enumerate(
                    tf.data.Dataset.from_tensor_slices((test_kg.h, test_kg.r, test_kg.t)).batch(test_batch_size)):

                if target_entity_type == "tail":
                    batch_source = batch_h
                    batch_target = batch_t
                else:
                    batch_source = batch_t
                    batch_target = batch_h

                translated = model([batch_source, batch_r], target_entity_type=target_entity_type)

                tiled_entity_embeddings = tf.tile(tf.expand_dims(normed_entity_embeddings, axis=0),
                                                  [batch_h.shape[0], 1, 1])
                tiled_translated = tf.tile(tf.expand_dims(translated, axis=1),
                                           [1, normed_entity_embeddings.shape[0], 1])
                dis = compute_distance(tiled_translated, tiled_entity_embeddings)

                ranks = tf.argsort(tf.argsort(dis, axis=1), axis=1).numpy()
                target_ranks = ranks[np.arange(len(batch_target)), batch_target.numpy()]
                
                
                for target_rank in target_ranks:
                    if target_rank == 0:
                        mrr.append(0)
                    else:
                        mrr.append(1.0 / target_rank)
                        
                mean_ranks.extend(target_ranks)
                
                avg_count = np.mean((target_ranks <= 10))
                hits_at_ten.append(avg_count)
                avg_count = np.mean((target_ranks <= 3))
                hits_at_three.append(avg_count)
                avg_count = np.mean((target_ranks <= 1))
                hits_at_one.append(avg_count)

        print("epoch = {}\tmean_rank = {}\tmrr = {}".format(epoch, np.mean(mean_ranks), np.mean(mrr)))
        print("Hits @ 10: {:.6f}, Hits @ 3: {:.6f}, Hits @ 1: {:.6f}".format(np.mean(hits_at_ten), np.mean(hits_at_three), np.mean(hits_at_one)))
        if(np.mean(mrr) > best_mrr_module):
            best_mrr_module = np.mean(mrr)
            best_mr_module = np.mean(mean_ranks)
            best_module_hits_ten = np.mean(hits_at_ten)
            best_module_hits_three = np.mean(hits_at_three)
            best_module_hits_one = np.mean(hits_at_one)
            best_module_epoch = epoch
            print(model.relation_embeddings.numpy())
            np.savetxt('entity_vec_FB15k_237.txt', )
#             with open('entity_vec_FB15k_237.txt', 'w') as f:
#                 for num in np.nditer(model.relation_embeddings.numpy()):
#                     print(num)
#                     f.write(num + '\t')
#             f.close()
#             with open('relation_vec_FB15k_237.txt', 'w') as f:
#                 for line in model.relation_embeddings.numpy():
#                     f.write(line + '\n')
#             f.close()
        print("best_epoch = {}\tbest_mean_rank = {}\tbest_mrr = {}".format(best_module_epoch, best_mr_module, best_mrr_module))
        print("best_Hits @ 10: {:.6f}, best_Hits @ 3: {:.6f}, best_Hits @ 1: {:.6f}".format(best_module_hits_ten, best_module_hits_three, best_module_hits_one))

files.download('entity_vec_FB15k_237.txt')
# files.download('relation_vec_FB15k_237.txt')
