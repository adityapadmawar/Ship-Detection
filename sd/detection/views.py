from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.shortcuts import get_object_or_404, render
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

# Create your views here.
from .forms import CreateUserForm

def index(request):
    return render(request, "index.html")

def methods(request):
    return render(request, "methods.html")

def register(request):
    if request.user.is_authenticated:
        return redirect('upload')
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Account was created for ' + user)
                return redirect('login')

        context = {'form':form}
        return render(request, "register.html", context)

def loginPage(request):
    if request.user.is_authenticated:
        return redirect('upload')
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)

            if user is not None:
                login(request, user)
                return redirect('upload')
            else:
                messages.info(request, 'Username or Pssword is incorrect')

        context = {}
        return render(request, "login.html", context)

def logoutUser(request):
    logout(request)
    return redirect('login')


@login_required(login_url='login')
def upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)

    #<-- print the latest file added -->    
    import glob
    import os

    list_of_files = glob.glob('media/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)

    #<-- end -->

    return render(request, "upload.html", context)



@login_required(login_url='login')
def detect(request):
    if request.method == 'POST':


    
        #<!-- ship detection code start -->

        import numpy as np # linear algebra
        import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
        import os
        import time
        import sys
        import mrcnn.model as modellib
        from mrcnn.config import Config
        from mrcnn.model import log
        from mrcnn import utils
        from mrcnn import visualize
        from skimage.morphology import label
        from skimage.data import imread
        import glob
        import os
        import matplotlib
        # Agg backend runs without a display
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from keras import backend as K


        list_of_files = glob.glob('media/*') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        TRAINING_VALIDATION_RATIO = 0.2
        WORKING_DIR = './'
        LOGS_DIR = os.path.join(WORKING_DIR, "logs")
        SHIP_CLASS_NAME = 'ship'
        IMAGE_WIDTH = 768
        IMAGE_HEIGHT = 768
        SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)

        # if to clone Mask_R-CNN git when it exists 
        UPDATE_MASK_RCNN = False


        class ShipDetection(utils.Dataset):
            """Ship Detection Dataset
            """
            def __init__(self, image_file_dir, ids, masks, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
                super().__init__(self)
                self.image_file_dir = image_file_dir
                self.ids = ids
                self.masks = masks
                self.image_width = image_width
                self.image_height = image_height
                
                # Add classes
                self.add_class(SHIP_CLASS_NAME, 1, SHIP_CLASS_NAME)
                self.load_dataset()
                
            def load_dataset(self):
                """Load dataset from the path
                """
                # Add images
                for index, row in self.ids.iterrows():
                    image_id = row['ImageId']
                    image_path = os.path.join(self.image_file_dir, image_id)
                    rle_mask_list = row['RleMaskList']
                    #print(rle_mask_list)
                    self.add_image(
                        SHIP_CLASS_NAME,
                        image_id=image_id,
                        path=image_path,
                        width=self.image_width, height=self.image_height,
                        rle_mask_list=rle_mask_list)

            def load_mask(self, image_id):
                """Generate instance masks for shapes of the given image ID.
                """
                info = self.image_info[image_id]
                rle_mask_list = info['rle_mask_list']
                mask_count = len(rle_mask_list)
                mask = np.zeros([info['height'], info['width'], mask_count], dtype=np.uint8)
                i = 0
                for rel in rle_mask_list:
                    if isinstance(rel, str):
                        np.copyto(mask[:,:,i], rle_decode(rel))
                    i += 1
                
                # Return mask, and array of class IDs of each instance. Since we have
                # one class ID only, we return an array of 1s
                return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
            
            def image_reference(self, image_id):
                """Return the path of the image."""
                info = self.image_info[image_id]
                if info['source'] == SHIP_CLASS_NAME:
                    return info['path']
                else:
                    super(self.__class__, self).image_reference(image_id)

        class ShipDetectionGPUConfig(Config):
            """
            Configuration of Ship Detection Dataset 
            Overrides values in the base Config class.
            From https://github.com/samlin001/Mask_RCNN/blob/master/mrcnn/config.py
            """
            # https://www.kaggle.com/docs/kernels#technical-specifications
            NAME = 'ASDC_GPU'
            # NUMBER OF GPUs to use.
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
            
            NUM_CLASSES = 2  # ship or background
            IMAGE_MIN_DIM = IMAGE_WIDTH
            IMAGE_MAX_DIM = IMAGE_WIDTH
            STEPS_PER_EPOCH = 300
            VALIDATION_STEPS = 50
            SAVE_BEST_ONLY = True
            
            # Minimum probability value to accept a detected instance
            # ROIs below this threshold are skipped
            DETECTION_MIN_CONFIDENCE = 0.94

            # Non-maximum suppression threshold for detection
            # Keep it small to merge overlapping ROIs 
            DETECTION_NMS_THRESHOLD = 0.05

            
        class InferenceConfig(ShipDetectionGPUConfig):
            GPU_COUNT = 1
            # 1 image for inference 
            IMAGES_PER_GPU = 1

        inference_config = InferenceConfig()

        # create a model in inference mode
        infer_model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=WORKING_DIR)

        #model_path = infer_model.find_last()
        model_path = 'detection/mask_rcnn_asdc_gpu_0008.h5'
        # Load trained weights
        print("Loading weights from ", model_path)
        infer_model.load_weights(model_path, by_name=True)


        image_id = latest_file # using the last uploaded image for detection
    

        original_image=mpimg.imread(image_id)

        results = infer_model.detect([original_image], verbose=1)

        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], ['BG', 'ship'], r['scores'])
        plt.savefig('static/output/detectedship.jpg',bbox_inches='tight')
        plt.close()

        #<!-- ship detection code end -->
        K.clear_session()


    return render(request,"detect.html")


@login_required(login_url='login')
def image(request):
    return render(request, "image.html")    