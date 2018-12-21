'''
Training file for multi kernel CNN model we built for msaic submission
'''

# dependencies
import argparse # argparse
import numpy as np # linear algebra
import os # OS
from glob import glob # file handling

# custom model
from multi_cnn_msaic_model import multi_kernel_cnn, get_data_train


EMBEDDING_DIMS = 50
MAX_LENGTH_QUERY = 12
MAX_LENGTH_PASSAGE = 80
MAX_LENGTH_TOTAL = MAX_LENGTH_QUERY + MAX_LENGTH_PASSAGE
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 128
DROP_OUT = 0.3

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--qpl-file', type = str, help = 'path to numpy dumps')
	parser.add_argument('--emb-file', type = str, help = 'path to embedding matrix')
	parser.add_argument('--save-folder', type = str, default = '../saved_model', help = 'path to folder where saving model')
	parser.add_argument('--num-epochs', type = int, default = 5, help = 'number fo epochs for training')
	parser.add_argument('--model-name', type = str, default = 'multi_cnn', help = 'name of the model')
	parser.add_argument('--val-split', type = float, default = 0.1, help = 'validation split ratio')
	parser.add_argument('--save-frequency', type = int, default = 2, help = 'save model after these steps')
	parser.add_argument('--log-path', type = str, default = ',../terminal_log/log_multi_cnn.log', help = 'folder to save terminal out log')
	parser.add_argument('--batch-size', type = int, default = 512, help = 'size of minibatch')
	parser.add_argument('--thresh-upper', type = float, default = 0.9, help = 'upper threshold for dummy accuracy check')
	parser.add_argument('--thresh-lower', type = float, default = 0.2, help = 'lower threshold for dummy accuracy check')
	args = parser.parse_args()

	'''
	Step 1: Before the models is built and setup, do the preliminary work
	'''

	# make the folders
	if not os.path.exists(args.save_folder):
		os.makedirs(args.save_folder)
		print('[*] Model saving folder could not be found, making folder ', args.save_folder)

	# We need to get the list of all the q, p and l files that were generated
	q_paths = sorted(glob(args.qpl_file + '_q*.npy'))
	p_paths = sorted(glob(args.qpl_file + '_p*.npy'))
	l_paths = sorted(glob(args.qpl_file + '_l*.npy'))

	print(q_paths)
	print(p_paths)
	print(l_paths)

	'''
	Step 2: All the checks are done, make the model
	'''
	print('[*] Data loading ...')
	# load the training numpy matrix
	for i in range(len(q_paths)):
		print('... loading file number:', i)
		if i == 0:
			train_q = np.load(q_paths[i])
			train_p = np.load(p_paths[i])
			train_l = np.load(l_paths[i])
		else:
			q_ = np.load(q_paths[i])
			p_ = np.load(p_paths[i])
			l_ = np.load(l_paths[i])
			train_q = np.concatenate([train_q, q_])
			train_p = np.concatenate([train_p, p_])
			train_l = np.concatenate([train_l, l_])
	
	# load embedding matrix
	print('... loading embedding matrix')
	embedding_matrix = np.load(args.emb_file)

	print('[*] ... Data loading complete!')

	# load the model, this is one line that will be changed for each case
	print('[*] Making model')
	model = multi_kernel_cnn(maxLengthTotal = MAX_LENGTH_TOTAL,
							embedDims = EMBEDDING_DIMS, embedMatrix = embedding_matrix,
							batchSize = args.batch-size,
							filterSizes = FILTER_SIZES,
							numFilters = NUM_FILTERS)

	print('model built successfully, loading train data...')

	'''
	Step 3: Train the model
	'''

	# creating object of get data class
	dataTrain = get_data_train(queriesData = train_q,
							passagesData = train_p,
							labelsData = train_l,
							batchSize = args.batch-size,
							maxLengthQuery = MAX_LENGTH_QUERY,
							maxLengthPassage = MAX_LENGTH_PASSAGE,
							maxLengthTotal = MAX_LENGTH_TOTAL,
							padId = len(embedding_matrix))
	print('train data loaded successfully, going into training...')
	# shuffling data
	dataTrain.shuffle_data()

	# creating session to run training
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	#merge = tf.summary.merge_all()

	with tf.Session() as sess:

	sess.run(init)
	#trainWriter = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
	#testWriter = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)

	for i in range(EPOCHS):

		batchX, batchY = cnn_model.get_next_batch()
		sess.run(model.train, feed_dict = {model.x : batchX, model.y : batchY, model.keepProb : DROP_OUT})
		saver.save(sess, os.path.join(args.save-folder, args.model-name), global_step = args.save-frequency)

		if i % 2 == 0:
			# calculating train accuracy
			acc, lossTmp = sess.run([model.accuracy, model.loss], feed_dict = {model.x : batchX, model.y : batchY, model.keepProb : DROP_OUT})
			toFile = 'Iter: '+str(i)+' Minibatch_Loss: '+"{:.6f}".format(lossTmp)+' Train_acc: '+"{:.5f}".format(acc)+"\n"
			print(toFile)
			with open(args.log-path, 'a') as f:
				f.write(toFile)
