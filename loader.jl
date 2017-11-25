function data_loader(max_training_set)
    ArrVersion=readdlm("train_test.csv", ',')
    ArrVersion=ArrVersion[2:301,1:3073]
    ArrVersion = convert(Array{Float32},ArrVersion)
    ######Getting the required values for training

    train_features = ArrVersion[1:max_training_set,1:3072]
    train_labels = ArrVersion[1:max_training_set,3073:3073]
    ######Getting the required values for testing
    test_features = ArrVersion[241:300,1:3072]
    test_labels = ArrVersion[241:300,3073]
    ####### freeing Memerory space ArrVesion reference #################
    ArrVersion = 0

    ####### Returning A dictionary containing DataSet
    dataset = Dict(
    "train_features"=> train_features,
    "train_labels"=> train_labels,
    "images_test"=> test_features,
    "labels_test"=> test_labels)
    
    return dataset
end

function trainloader()
    d = data_loader(240)
    train = reshape(d["train_features"],(240,3,32,32))#tranform to 240,3,32,32) 
    images = permutedims(train,[1,3,4,2]) #then to (240,32,32,3)
    #println(size(images),size(train) )
    labels = d["train_labels"]
    d = 0;train=0
    return images,labels
end

function testloader()
    d = data_loader(240)
    test = reshape(d["images_test"],(60,3,32,32))#tranform to 60,3,32,32) 
    images = permutedims(test,[1,3,4,2]) #then move color channels to the last (240,32,32,3)
    #println(size(images),size(train) )
    labels = d["labels_test"]
    d = 0
    test=0
    return images,labels
end
#x,y = testloader()
#println(size(x),size(y))

function batching(batch_size=16)
    # Number of images in the training-set. = 240
    if batch_size > 240
        batch_size = 16
    end
    x,y = trainloader()
    num_images = 240
 
    # Create a random index.
    idx = rand(1:240,batch_size,1) #-->5-element Array{Int64,1} eg.232,234,5,7,55,... 
                                        #but rand(1:num_images,batch_size) 5X1-element Array{Int64,2}: eg.232,234,5,7,55,...
    
    # Create 2-arrays of batch_size X 3072, and batch_size X 1
    x_batch = zeros(Float32,16,32,32,3)
    y_batch = zeros(Float32,16,)
    
    
    index = 1
    for k in idx
        #Copy to every row k in x_batch and y_batch
        x_batch[index:index,:,:,:] = x[k:k,:,:,:]
        y_batch[index:index,] = y[k:k,] # check to confirm are copied
        index = index + 1
    end 
    y=0
    x=0
return x_batch,reshape(y_batch,(16,1)) 
end




function next_batch(batch_size=16)
    # Number of images in the training-set. = 240
    if batch_size > 240
        batch_size = 16
    end
    
    num_images = 240
    dataset = data_loader(240)
    images_train = dataset["train_features"] # println(size(dataset["train_features"]))--> (240, 3072)
    labels_train = dataset["train_labels"] 
    dataset = 0
    # Create a random index.
    idx = rand(1:num_images,batch_size,1) #-->5-element Array{Int64,1} eg.232,234,5,7,55 
                                        #but rand(1:num_images,batch_size) 5X1-element Array{Int64,2}: eg.232,234,5,7,55
    
    # Create 2-arrays of batch_size X 3072, and batch_size X 1
    x_batch = zeros(Float32,batch_size,3072)
    y_batch = zeros(Float32,batch_size,1)
    
    
    index = 1
    for k in idx
        #Copy to every row k in x_batch and y_batch
        x_batch[index:index,1:3072] = images_train[k:k,1:3072]
        y_batch[index:index,] = labels_train[k:k,] # check to confirm are copied
        index = index + 1
    end
    
#     # Use the random index to select random images and labels.
#     x_batch = images_train[idx, :, :, :] #x_batch = images_train[idx, :, :, :]
#     y_batch = labels_train[idx, :]       #y_batch = labels_train[idx, :]
    #x_batch = 

return x_batch, y_batch 
end

function load_test_set()
    dataset = data_loader(240)
    images_test = dataset["images_test"]
    labels_test = dataset["labels_test"]
    dataset = 0 
    
return images_test,reshape(labels_test(60,1))
end
#x,y = load_test_set()
#println( size(x),y)    

