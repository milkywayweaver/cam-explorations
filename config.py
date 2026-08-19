CONFIG = {
    'run_name':'EfficientNet B0 - Batch Size 32', # String
    'backbone':'effnet', # Used in script.py -- String: "vgg", "resnet", "effnet"
    'seed':42, # Used in script.py # Int
    'augment':[None], # Used in augment.py -- List of strings: "geometric", "color", "blur", "erasing"
    'batch_size':32, # Used in script.py # Int
    'relu':True, # Used in trainloop.py -- # Bool
    'ch_project':'mapper' # Used in script.py # String: "mapper", "duplicate"
}