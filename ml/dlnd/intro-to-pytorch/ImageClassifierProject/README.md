# Project - Create Your Own Image Classifier

The `train.py` file accepts command line arguments which have
 been slightly modified from the exact requirements laid out in the
  rubric. The changes are as follows:
  
  - The training dataset has to be now supplied as a command line
   argument via the `--data_dir` tag.
   
  - The `hidden_units` tag has been changed to be a list of nodes
   in the hidden layers instead of a simple number. So now, it has to
    be supplied as a string encoded list to the command line as
     follows: 
     
     `--hidden_units "[512, 256]"` 
  - The choice of which device (CUDA or CPU) to run the training on
   has been changed from the `--gpu` tag for the command line to
    the `--device` tag which requires an argument. Therefore, if
     you want to run the training on the GPU, this argument has to
      be supplied as `--device cuda`
      
  - Currently only 3 architectures are implemented, namely `alexnet
  `, `densenet121` and `vgg16`.  
  
  Example input:
  
  `python3 train.py --device cuda --arch vgg16 --learning_rate 0.001 --hidden_units "[512, 256]" --epochs 3`  

In a similar fashion the `predict.py` file does not strictly
 conform to guidelines laid out in the project instructions.
 
   - The choice of which device (CUDA or CPU) to run the prediction on
   has been changed from the `--gpu` tag for the command line to
    the `--device` tag which requires an argument. Therefore, if
     you want to run the training on the GPU, this argument has to
      be supplied as `--device cuda`
      
   - The `--image` and `--checkpoint` command line arguments have
    been converted to required arguments and these represent full
     paths to the image and the checkpoint respectively.
     
   - The prediction result is returned as a dictionary of the top n
    classes of prediction as chosen by the user, where the keys are
     the class names and values are dictionaries of respective
      class labels and class probabilities. 
      
  Example input:
  
  `python3 predict.py --image /home/mhasan3/Desktop
  /ImageClassifierProject/flowers/test/17/image_03911.jpg
   --checkpoint /home/mhasan3/checkpoint.pth --topk 3 --device cuda
   `     
     
     
       
 
 
      
       