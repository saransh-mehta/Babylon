'''
Here we are building CNN architecture for text classification
We have already saved 3D volume of the sentences with median length as max_length,
with the word2vec trained embeddings with gensim,
Hence we have (max_length, Embed_dims, 1)

The CNN mode architecture is:
The first layers embeds words into low-dimensional vectors.
The next layer performs convolutions over the embedded word vectors
using multiple filter sizes. For example, sliding over 3, 4 or 5 words at a time.
Next, we max-pool the result of the convolutional layer into a long feature vector,
add dropout regularization, and classify the result using a softmax layer.

improvements:-

1. Reduce batch size to around 64
2. Add one more dense layer and gradually bring down
3. Increase number of filters

 '''

class multi_kernel_cnn:
	def __init__(self, maxLengthTotal, embedDims, embedMatrix, batchSize, filterSizes, numFilters):

		self.maxLengthTotal = maxLengthTotal # maximum seq length after concatenating query and pass together
		self.embedDims = embedDims # dimensions of embedding - 50
		self.embedMatrix = embedMatrix #glove embedding matrix of size (Vocab_size, embed_dims)
		self.batchSize = batchSize
		self.filterSizes = filterSizes # different filter sizes for multi-kernel CNN
		self.numFilters = numFilters #no. of feature maps (filters)


 		# we will write the model in init so that we can simply call
 		# the class to initialize the graph

 		#placeholders
 		with tf.name_scope('inp_placeholders') as scope:

 			self.x = tf.placeholder(shape = [self.batchSize, self.maxLength],
 								name = 'inp_x', dtype = tf.int32)
 			self.y = tf.placeholder(shape = [self.batchSize, 1], name = 'target_y',
 								dtype = tf.int32)
 			self.keepProb = tf.placeholder(dtype = tf.float32, name='dropout')
 			#dropout ratio has to be a placeholder
 			# which will be fed value at training like x

 			self.embedding_matrix = tf.constant(emb, name = 'embedding_matrix', dtype = tf.float32)

 		with tf.name_scope('embedding') as scope:
 			xEmbed = tf.nn.embedding_lookup(self.embedding_matrix, self.x)

 		pooledOutputList = []
 		for i, filterSize in enumerate(filterSizes):

 			with tf.name_scope('CNN_filter_size_%s'%filterSize) as scope:

 				# as filter shape also needs to be a tensor
 				filterr = tf.Variable(tf.truncated_normal([filterSize, 
 														  embedDims, 1, numFilters],
 														  stddev=0.1),name='filter')
 				conv1 = tf.nn.conv2d(
 					self.x, filter=filterr,strides=[1,1,1,1],padding='VALID',name='conv1')
 				# as in our model our kernal size would convolute the 
 				# whole embedding for a word.

 				#convolving with VALID padding means, we slide without adding
 				#padding to the edges, hence we get the output dimension as
 				# [1, maxLength - filterSize +1, 1,1]

 				#adding activation and bias for non-linearity
 				b = tf.Variable(tf.constant(0.1, shape=[numFilters],name='bias'))
 				activation1 = tf.nn.relu(
 					tf.nn.bias_add(conv1, b), name='relu_actv')

 				# maxpool
 				maxPooled = tf.nn.max_pool(
 					activation1, ksize= [1, maxLength - filterSize +1, 1,1],
 					strides = [1,1,1,1], padding='VALID', name='max_pool1')
 				#Performing max-pooling over the output of a specific filter size
 				#leaves us with a tensor of shape [batch_size, 1, 1, num_filters].
 				#This is essentially a feature vector, where the last dimension 
 				#corresponds to our features

 				# here we need to combine the maxPooled output for all the 
 				# different filter size that we use, hence we will apend the
 				# output into a list
 				pooledOutputList.append(maxPooled)

 		#combining the pooled output for different filters
 		self.finalPoolOut = tf.concat(pooledOutputList, axis = 3)
 		totalFiltersNum = numFilters * len(filterSizes)
 		self.finalPoolOutFlatten = tf.reshape(self.finalPoolOut, [-1,totalFiltersNum])

 		#dropout in fully connected layer
 		with tf.name_scope('dropout') as scope:
 			self.dropoutLayer = tf.nn.dropout(self.finalPoolOutFlatten, self.keepProb)

 		with tf.name_scope('out_layer') as scope:
 			self.finalOut = tf.layers.dense(self.dropoutLayer,units=classNum,name='final_output')
 			self.predictions = tf.argmax(self.finalOut, 1, name='predictions')
 		
 		with tf.name_scope('loss') as scope:
 			losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.finalOut,labels=self.y))
 			self.loss = tf.reduce_mean(losses)

 		with tf.name_scope('train') as  scope:
 			optimizer = tf.train.AdamOptimizer()
 			self.train = optimizer.minimize(self.loss)
 		
 		with tf.name_scope('accuracy') as scope:
 			correctPrediction = tf.equal(tf.argmax(self.finalOut, axis = 1), tf.argmax(self.y, axis = 1))
 			self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32)) * 100

 		with tf.name_scope('confusion_matrix') as scope:
 			self.confMatrix = tf.confusion_matrix(labels=tf.argmax(self.y,axis=1), predictions=tf.argmax(self.finalOut, axis = 1), num_classes=classNum)



class get_data_train:
	def __init__(self, queriesData, passagesData, labelsData,
				 batchSize, maxLengthQuery, maxLengthPassage, maxLengthTotal, padId, seed=2):
		
		self.queriesData = queriesData
		self.passagesData = passagesData
		self.labelsData = labelsData
		self.batchSize = batchSize
		self.seed = seed
		self.maxLengthQuery = maxLengthQuery
		self.maxLengthPassage = maxLengthPassage
		self.maxLengthTotal = maxLengthTotal


	def shuffle_data(self):
		from sklearn.utils import shuffle
		self.queriesData,self.passagesData,self.labelsData = shuffle(self.queriesData,self.passagesData,self.labelsData, random_state = self.seed)

	def length_clip(self, inpSeq, length):

		'''
		First, we also need to check if the length of query is exceeding maximum query length,
		and if so need to clip extra part. Same goes with passage.
		This function performs it on provided input sequence
		'''
		if len(inpSeq) > length:
			inpSeq = inpSeq[:length]

		# here we don't add padding for length less than max length
		return inpSeq

	def concatenate_query_passage_and_pad(self, query, passage, maxLength, padId):

		'''
		Here a query and it's respective passage needs to be passed,
		and this function concatenates them  horizontally to make one sentence.
		This function also checks if the total concatenation is greater than maximum
		length, then clips off extra length, and if less than maximum length,
		adds padId at last (index of <PAD> word in embedding).
		'''
		total = query + passage

		if len(total) > maxLength:
			#print('txtVec length greater')
			total = total[:length]

		if len(total) < maxLength:
			#print('txtVec length smaller')
			#padId = np.zeros((1,EMBED_DIMS, 1))
			for i in range(maxLength - len(total)):
				#txtVec = np.vstack((txtVec, padVec))
				total.append(padId)

		# here there is no condtion for len(txtVec) == length,
		# hence, they will b passed as it is,

		return total


	def get_next_batch(self):
		# this fn will take out batches of batchSize from training data
		indexes = list(range(len(self.queriesData)))
		np.random.shuffle(indexes)
		batch = indexes[:self.batchSize]
		# now the trick is to convert the words into their respective integer through
		# wordIndexMap and then feed into Rnn
		batchX = []
		batchY = []
		for i in batch:
			query = length_clip(inpSeq = self.queriesData[i], length = self.maxLengthQuery)
			passage = length_clip(inpSeq=self.passagesData, length=self.maxLengthPassage)

			x = concatenate_query_passage_and_pad(query, passage, self.maxLengthTotal, self.padId)
			batchX.append(x)
			batchY.append(self.labelsData[i])

		return batchX, batchY

