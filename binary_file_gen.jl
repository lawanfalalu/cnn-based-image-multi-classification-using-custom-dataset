using DataFrames
using ImageMagick
using Images
function generate()
	path = "/home/spg/train/list_label_digits.csv" # setting path image names list
	img_path = "/home/spg/train/IMG32X32" #setting a path for images directory
	data = readtable(path,separator=',') # importing list as an array I called data
	image_size = 32 * 32 # resulting to 32X32(RGB)

	X = zeros(Float64,size(data)[1], 3072)
	y = zeros(Int8, size(data)[1], 1)

	for (index, b) in enumerate(data[1]);
	    theImage = load(img_path*'/'*b) #loading the image using load instead of using imread()-deprecated
	    red_channel =reshape(map(ColorTypes.red, theImage), 1,image_size) # vectorizing the  R-property
	    green_channel =reshape(map(ColorTypes.green, theImage), 1,image_size) # vectorizing the  G-property
	    blue_channel =reshape(map(ColorTypes.blue, theImage), 1,image_size) # vectorizing the  B-property
	    X[index, :] = [red_channel  green_channel  blue_channel]
	end

	for (index, b) in enumerate(data[2]);
	    y[index] = b
	end


	# the train.csv having (300, 3073)
	df = DataFrame([X y])
	writetable("train_test_gedit.csv", df) #creates a csv file and saves the data, df
	# writetable("labels_data.csv",DataFrame(y)) # 300 X 3072 dims
	# writetable("images_data.csv",DataFrame(X)) # 300 X 1 dims
end

