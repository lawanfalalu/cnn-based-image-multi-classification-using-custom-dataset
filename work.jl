using TensorFlow
sess = tf.Session()
reader = TensorFlow.TextLineReader()
training_set = TensorFlow.train.String_input_producer(["IMG32X32.cvs"])
img, lbl = reader.read(training_set)
with Session as sess
	coord = TensorFlow.trainCoordinator()



