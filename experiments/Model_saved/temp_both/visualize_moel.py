import tensorflow as tf
import tensorboard as tb
g = tf.Graph() 

with g.as_default() as g: 
    tf.train.import_meta_graph('/Users/summerxia/Desktop/maddpg-TF/experiments/temp_only_obstacle/policy.meta') 

with tf.Session(graph=g) as sess: 
    file_writer =tf.summary.FileWriter("logs/test", sess.graph)

# for event in tf.train.summary_iterator('/Users/summerxia/Desktop/maddpg-TF/experiments/temp_both/logs/my-model/events.out.tfevents.1546884667.edu'):
#     for v in e.summary.value:
#         if v.tag == 'loss' or v.tag == 'accuracy':
#             print(v.simple_value)