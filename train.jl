using TensorFlow
using Distributions
include("loader.jl") 

# dataset = data_loader(240)
# images_train = dataset["train_features"] # println(size(dataset["train_features"]))--> (240, 3072)
# labels_train = dataset["train_labels"]   # println(size(dataset["train_labels"]))--> (240, 1)
# images_test = dataset["images_test"]     # println(size(dataset["images_test"])) --> (60, 3072)
# labels_test = dataset["labels_test"]     # println(size(dataset["labels_test"])) --> (60, 1)
# ########## Freeing space of dataset #################
# dataset = 0

session = Session(Graph())
function weight_variable(shape)
    initial = map(Float32, rand(Normal(0, .001), shape...))
    return Variable(initial)
end

function bias_variable(shape)
    initial = fill(Float32(.1), shape...)
    return Variable(initial)
end

function conv2d(x, W)
    nn.conv2d(x, W, [1, 1, 1, 1], "SAME")
end

function max_pool_2x2(x)
    nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
end

x = placeholder(Float32)
y_ = placeholder(Float32)

W_conv1 = weight_variable([5, 5, 3, 32]) 
b_conv1 = bias_variable([32])

x_image = reshape(x, [-1, 32, 32, 3])

h_conv1 = nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = reshape(h_pool2, [-1, 7*7*64])
h_fc1 = nn.relu(h_pool2_flat * W_fc1 + b_fc1)

keep_prob = placeholder(Float32)
h_fc1_drop = nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 9])
b_fc2 = bias_variable([9])

y_conv = nn.softmax(h_fc1_drop * W_fc2 + b_fc2)


cross_entropy = reduce_mean(-reduce_sum(y_ .* log(y_conv), reduction_indices=[2]))

train_step = train.minimize(train.AdamOptimizer(1e-4), cross_entropy)

correct_prediction = indmax(y_conv, 2) .== indmax(y_, 2)

accuracy = reduce_mean(cast(correct_prediction, Float32))

run(session, initialize_all_variables())

for i in 1:40
    images_train,labels_train = next_batch(16) # randomly generate batches from training dataset 
    if i%4 == 1
        train_accuracy = run(session, accuracy, Dict(x=>images_train, y_=>labels_train, keep_prob=>1.0))
        info("step $i, training accuracy $train_accuracy")
    end
    run(session, train_step, Dict(x=>images_train, y_=>labels_train, keep_prob=>.5))
end

images_test, labels_test = load_test_set() # 60 X 3072, 60 X 1 Arrays

test_accuracy = run(session, accuracy, Dict(x=>images_test, y_=>labels_test, keep_prob=>1.0))
info("test accuracy $test_accuracy")
