import cv2
import imageio
import argparse
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from albumentations import Blur, CLAHE, ChannelDropout, ChannelShuffle, \
                           CoarseDropout, Downscale, Equalize, FancyPCA, \
                           FromFloat, GaussNoise, GaussianBlur, GlassBlur, \
                           HueSaturationValue, IAAAdditiveGaussianNoise, \
                           IAAEmboss, IAASharpen, IAASuperpixels, ISONoise, \
                           ImageCompression, InvertImg, MedianBlur, MotionBlur,\
                           MultiplicativeNoise, Normalize, Posterize, RGBShift,\
                           RandomBrightnessContrast, RandomFog, RandomGamma,\
                           RandomRain, RandomShadow, RandomSnow, \
                           RandomSunFlare, Solarize, ToFloat, ToGray, ToSepia
from albumentations import CenterCrop, Crop, CropNonEmptyMaskIfExists, \
                           ElasticTransform, Flip, GridDistortion, GridDropout,\
                           HorizontalFlip, IAAAffine, IAACropAndPad, \
                           IAAPiecewiseAffine, LongestMaxSize, \
                           OpticalDistortion, RandomCrop, RandomGridShuffle, \
                           RandomResizedCrop, RandomRotate90, RandomScale, \
                           RandomSizedCrop, Resize, Rotate, ShiftScaleRotate, \
                           SmallestMaxSize, Transpose, VerticalFlip


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, dest='image_path', 
                        action='store', required=True, help='Path to an image.')
    parser.add_argument('--augmentation', type=str, dest='augmentation', 
                        action='store', required=True, 
                        help='Augmentation to be applied in the image.')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    image_path = args.image_path
    augmentation = args.augmentation

    image = cv2.imread(image_path)
    height, width, channels = image.shape 
    image_name = image_path.split('/')[-1]

    # Pixel Level transforms:

    ## Arithmetic

    if augmentation == 'add':
        transform = iaa.Add((-75, 75))
        transformed_image = transform(image=image)
    
    elif augmentation == 'add_elementwise':
        transform = iaa.AddElementwise((-75, 75))
        transformed_image = transform(image=image)

    elif augmentation == 'additive_gaussian_noise':
        transform = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
        transformed_image = transform(image=image)
    
    elif augmentation == 'additive_laplace_noise':
        transform = iaa.AdditiveLaplaceNoise(scale=(0, 0.2*255))
        transformed_image = transform(image=image)

    elif augmentation == 'additive_poisson_noise':
        transform = iaa.AdditivePoissonNoise(lam=(0, 40))
        transformed_image = transform(image=image)

    elif augmentation == 'multiply':
        transform = iaa.Multiply((0.1, 2.0))
        transformed_image = transform(image=image)

    elif augmentation == 'multiply_elementwise':
        transform = iaa.MultiplyElementwise((0.1, 2.0))
        transformed_image = transform(image=image)

    elif augmentation == 'dropout':
        transform = iaa.Dropout(p=(0, 0.2))
        transformed_image = transform(image=image)

    elif augmentation == 'coarse_dropout':
        transform = CoarseDropout(always_apply=True, max_holes=100)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'channel_droput':
        transform = ChannelDropout(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'salt':
        transform = iaa.Salt(0.1)
        transformed_image = transform(image=image)

    elif augmentation == 'coarse_salt':
        transform = iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1))
        transformed_image = transform(image=image)

    elif augmentation == 'pepper':
        transform = iaa.Pepper(0.1)
        transformed_image = transform(image=image)

    elif augmentation == 'coarse_pepper':
        transform = iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1))
        transformed_image = transform(image=image)

    elif augmentation == 'salt_and_papper':
        transform = iaa.SaltAndPepper(0.1)
        transformed_image = transform(image=image)

    elif augmentation == 'coarse_salt_and_papper':
        transform = iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1))
        transformed_image = transform(image=image)

    elif augmentation == 'impulse_noise':
        transform = iaa.ImpulseNoise(0.1)
        transformed_image = transform(image=image)

    elif augmentation == 'replace_elementwise':
        transform = iaa.ReplaceElementwise(0.1, [0, 255])
        transformed_image = transform(image=image)

    elif augmentation == 'cutout':
        transform = iaa.Cutout(nb_iterations=5)
        transformed_image = transform(image=image)

    elif augmentation == 'solarize':
        transform = Solarize(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'invert_img':
        transform = InvertImg(always_apply=True)
        transformed_image = transform(image=image)['image']

    ## Artistic

    elif augmentation == 'cartoon':
        transform = iaa.Cartoon(blur_ksize=3, segmentation_size=1.0,
                                saturation=2.0, edge_prevalence=1.0)
        transformed_image = transform(image=image)

    ## Blend

    elif augmentation == 'blend_alpha':
        transform = iaa.BlendAlpha(0.5, iaa.Grayscale(1.0))
        transformed_image = transform(image=image)

    elif augmentation == 'blend_alpha_simplex_noise':
        transform = iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0))
        transformed_image = transform(image=image)

    elif augmentation == 'blend_alpha_some_colors':
        transform = iaa.BlendAlphaSomeColors(iaa.Grayscale(1.0))
        transformed_image = transform(image=image)

    elif augmentation == 'blend_alpha_regular_grid':
        transform = iaa.BlendAlphaRegularGrid(nb_rows=(4, 6), nb_cols=(1, 4),
                                              foreground=iaa.Multiply(0.0))
        transformed_image = transform(image=image)

    elif augmentation == 'blend_alpha_mask':
        transform = iaa.BlendAlphaMask(iaa.InvertMaskGen(0.5, 
                                      iaa.VerticalLinearGradientMaskGen()),
                                      iaa.Clouds())
        transformed_image = transform(image=image)

    elif augmentation == 'blend_alpha_elementwise':
        transform = iaa.BlendAlphaElementwise(0.5, iaa.Grayscale(1.0))
        transformed_image = transform(image=image)

    elif augmentation == 'blend_alpha_vlg':
        transform = iaa.BlendAlphaVerticalLinearGradient(
                                                    iaa.AddToHue((-100, 100)))
        transformed_image = transform(image=image)

    elif augmentation == 'blend_alpha_hlg':
        transform = iaa.BlendAlphaHorizontalLinearGradient(
                                                    iaa.AddToHue((-100, 100)))
        transformed_image = transform(image=image)

    ## Blur

    elif augmentation == 'blur':
        transform = Blur(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'gaussian_blur':
        transform = GaussianBlur(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'median_blur':
        transform = MedianBlur(always_apply=True, blur_limit=(18, 25))
        transformed_image = transform(image=image)['image']

    elif augmentation == 'motion_blur':
        transform = iaa.MotionBlur(k=15)
        transformed_image = transform(image=image)

    elif augmentation == 'average_blur':
        transform = iaa.AverageBlur(k=(2, 11))
        transformed_image = transform(image=image)

    elif augmentation == 'bilateral_blur':
        transform = iaa.BilateralBlur(d=(3, 10), sigma_color=(250), 
                                                 sigma_space=(250))
        transformed_image = transform(image=image)

    elif augmentation == 'mean_shift_blur':
        transform = iaa.MeanShiftBlur()
        transformed_image = transform(image=image)

    elif augmentation == 'glass_blur':
        transform = GlassBlur(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'defocus_blur':
        transform = iaa.imgcorruptlike.DefocusBlur(severity=2)
        transformed_image = transform(image=image)

    elif augmentation == 'zoom_blur':
        transform = iaa.imgcorruptlike.ZoomBlur(severity=2)
        transformed_image = transform(image=image)

    ## Color
    elif augmentation == 'multiply_hue':
        transform = iaa.MultiplyHue((0.5, 1.5))
        transformed_image = transform(image=image)

    elif augmentation == 'addto_hue':
        transform = iaa.AddToHue((-100, 100))
        transformed_image = transform(image=image)
    
    elif augmentation == 'multiply_saturation':
        transform = iaa.MultiplySaturation((0.5, 1.5))
        transformed_image = transform(image=image)
    
    elif augmentation == 'addto_saturation':
        transform = iaa.AddToSaturation((-100, 100))
        transformed_image = transform(image=image)
    
    elif augmentation == 'saturate':
        transform = iaa.imgcorruptlike.Saturate(severity=5)
        transformed_image = transform(image=image)

    elif augmentation == 'remove_saturation':
        transform = iaa.RemoveSaturation()
        transformed_image = transform(image=image)
    
    elif augmentation == 'multiply_hue_and_saturation':
        transform = iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
        transformed_image = transform(image=image)

    elif augmentation == 'brightness_contrast':
        transform = RandomBrightnessContrast(always_apply=True, 
                                             brightness_limit=0.5)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'brightness':
        transform = iaa.imgcorruptlike.Brightness(severity=2)
        transformed_image = transform(image=image)

    elif augmentation == 'addto_hue_and_saturation':
        transform = iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
        transformed_image = transform(image=image)

    elif augmentation == 'hue_saturation':
        transform = HueSaturationValue(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'multiply_brightness':
        transform = iaa.MultiplyBrightness((0.1, 1.9))
        transformed_image = transform(image=image)

    elif augmentation == 'addto_brightness':
        transform = iaa.AddToBrightness((-50, 50))
        transformed_image = transform(image=image)

    elif augmentation == 'multiply_and_addtobrightness':
        transform = iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), 
                                                   add=(-30, 30))
        transformed_image = transform(image=image)

    elif augmentation == 'to_gray':
        transform = ToGray(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'posterize':
        transform = Posterize(always_apply=True, num_bits=2)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'to_sepia':
        transform = ToSepia(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'fancy_pca':
        transform = FancyPCA(always_apply=True, alpha=1.0)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'rgb_shift':
        transform = RGBShift(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'change_color_temperature':
        transform = iaa.ChangeColorTemperature((1100, 10000))
        transformed_image = transform(image=image)

    elif augmentation == 'kmeans_color_quantization':
        transform = iaa.KMeansColorQuantization()
        transformed_image = transform(image=image)

    elif augmentation == 'uniform_color_quantization':
        transform = iaa.UniformColorQuantization()
        transformed_image = transform(image=image)

    ## Contrast
    elif augmentation == 'contrast':
        transform = iaa.imgcorruptlike.Contrast(severity=2)
        transformed_image = transform(image=image)

    elif augmentation == 'clahe':
        transform = CLAHE(always_apply=True, tile_grid_size=(64, 64))
        transformed_image = transform(image=image)['image']

    elif augmentation == 'equalize':
        transform = Equalize(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'random_gamma':
        transform = RandomGamma(always_apply=True, gamma_limit=(120, 200))
        transformed_image = transform(image=image)['image']
    
    elif augmentation == 'gamma_contrast':
        transform = iaa.GammaContrast((2.0))
        transformed_image = transform(image=image)

    elif augmentation == 'sigmoid_contrast':
        transform = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
        transformed_image = transform(image=image)

    elif augmentation == 'log_contrast':
        transform = iaa.LogContrast(gain=(0.6, 1.4))
        transformed_image = transform(image=image)

    elif augmentation == 'linear_contrast':
        transform = iaa.LinearContrast((0.4, 1.6))
        transformed_image = transform(image=image)

    elif augmentation == 'histogram_equalization':
        transform = iaa.HistogramEqualization()
        transformed_image = transform(image=image)

    elif augmentation == 'all_channels_he':
        transform = iaa.AllChannelsHistogramEqualization()
        transformed_image = transform(image=image)

    elif augmentation == 'all_channels_clahe':
        transform = iaa.AllChannelsCLAHE()
        transformed_image = transform(image=image)

    ## Compression
    elif augmentation == 'image_compression':
        transform = ImageCompression(always_apply=True, quality_lower=10)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'downscale':
        transform = Downscale(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'pixelate':
        transform = iaa.imgcorruptlike.Pixelate(severity=4)
        transformed_image = transform(image=image)

    ## Convolutional
    elif augmentation == 'sharpen':
        transform = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
        transformed_image = transform(image=image)

    elif augmentation == 'emboss':
        transform = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
        transformed_image = transform(image=image)

    elif augmentation == 'edge_detect':
        transform = iaa.EdgeDetect(alpha=(0.0, 1.0))
        transformed_image = transform(image=image)

    elif augmentation == 'directed_edge_detect':
        transform = iaa.DirectedEdgeDetect(alpha=(1.0), 
                                           direction=(0.0, 1.0))
        transformed_image = transform(image=image)

    elif augmentation == 'convolve':
        matrix = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        transform = iaa.Convolve(matrix=matrix)
        transformed_image = transform(image=image)

    ## Corruption

    elif augmentation == 'gauss_noise':
        transform = GaussNoise(always_apply=True, var_limit=(200.0, 250.0))
        transformed_image = transform(image=image)['image']

    elif augmentation == 'multiplicative_noise':
        transform = MultiplicativeNoise(always_apply=True, 
                                        multiplier=(0.5, 1.5))
        transformed_image = transform(image=image)['image']

    elif augmentation == 'iso_noise':
        transform = ISONoise(always_apply=True, color_shift=(0.08, 0.1), 
                                                intensity=(0.5, 0.8))
        transformed_image = transform(image=image)['image']

    elif augmentation == 'shot_noise':
        transform = iaa.imgcorruptlike.ShotNoise(severity=2)
        transformed_image = transform(image=image)
    
    elif augmentation == 'speckle_noise':
        transform = iaa.imgcorruptlike.SpeckleNoise(severity=2)
        transformed_image = transform(image=image)
    
    elif augmentation == 'random_shadow':
        transform = RandomShadow(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'random_sun_flare':
        transform = RandomSunFlare(always_apply=True)
        transformed_image = transform(image=image)['image']
        
    elif augmentation == 'spatter':
        transform = iaa.imgcorruptlike.Spatter(severity=2)
        transformed_image = transform(image=image)

    elif augmentation == 'elastic_transform':
        transform = iaa.imgcorruptlike.ElasticTransform(severity=5)
        transformed_image = transform(image=image)
    
    ## Edges

    elif augmentation == 'canny':
        transform = iaa.Canny(alpha=(0.0, 0.9))
        transformed_image = transform(image=image)

    ## Pooling
    elif augmentation == 'average_pooling':
        transform = iaa.AveragePooling(5)
        transformed_image = transform(image=image)

    elif augmentation == 'max_pooling':
        transform = iaa.MaxPooling(5)
        transformed_image = transform(image=image)

    elif augmentation == 'min_pooling':
        transform = iaa.MinPooling(5)
        transformed_image = transform(image=image)

    elif augmentation == 'median_pooling':
        transform = iaa.MedianPooling(5)
        transformed_image = transform(image=image)


    ## Weather
    elif augmentation == 'fog':
        transform = iaa.imgcorruptlike.Fog(severity=2)
        transformed_image = transform(image=image)

    elif augmentation == 'random_rain':
        transform = RandomRain(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'snow':
        transform = iaa.imgcorruptlike.Snow(severity=2)
        transformed_image = transform(image=image)

    elif augmentation == 'frost':
        transform = iaa.imgcorruptlike.Frost(severity=2)
        transformed_image = transform(image=image)

    ####################
            
    elif augmentation == 'channel_shuffle':
        transform = ChannelShuffle(always_apply=True)
        transformed_image = transform(image=image)['image']
                        
    elif augmentation == 'from_float':
        transform = FromFloat(always_apply=True, max_value=127)
        transformed_image = transform(image=image)['image']
        
    elif augmentation == 'super_pixels':
        transform = IAASuperpixels(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'normalize':
        transform = Normalize(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'to_float':
        transform = ToFloat(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'center_crop':
        transform = CenterCrop(always_apply=True, width=400, height=400)
        transformed_image = transform(image=image)['image']
    
    elif augmentation == 'crop':
        transform = Crop(always_apply=True, x_max=400, y_max=400)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'crop_non_empty_mask':
        transform = CropNonEmptyMaskIfExists(always_apply=True, width=400, 
                                                                height=400)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'elastic_transform':
        transform = ElasticTransform(always_apply=True, alpha=10)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'flip':
        transform = Flip(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'grid_distortion':
        transform = GridDistortion(always_apply=True, distort_limit=0.5)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'grid_dropout':
        transform = GridDropout(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'horizontal_flip':
        transform = HorizontalFlip(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'iaaaffine':
        transform = IAAAffine(always_apply=True, p=1.0, rotate=180, shear=50)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'crop_and_pad':
        transform = IAACropAndPad(always_apply=True, percent=0.5)
        transformed_image = transform(image=image)['image']
    
    elif augmentation == 'longest_max_size':
        transform = LongestMaxSize(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'optical_distortion':
        transform = OpticalDistortion(always_apply=True, distort_limit=0.5)
        transformed_image = transform(image=image)['image']
    
    elif augmentation == 'random_crop':
        transform = RandomCrop(always_apply=True, width=200, height=200)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'random_grid_shuffle':
        transform = RandomGridShuffle(always_apply=True, grid=(5, 5))
        transformed_image = transform(image=image)['image']

    elif augmentation == 'random_resized_crop':
        transform = RandomResizedCrop(always_apply=True, width=100, height=100)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'random_rotate90':
        transform = RandomRotate90(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'random_scale':
        transform = RandomScale(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'random_sized_crop':
        transform = RandomSizedCrop(always_apply=True, height=500, width=500, 
                                                      min_max_height=[200, 200])
        transformed_image = transform(image=image)['image']

    elif augmentation == 'resize':
        transform = Resize(always_apply=True, height=500, width=500)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'rotate':
        transform = Rotate(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'shift_scale_rotate':
        transform = ShiftScaleRotate(always_apply=True, shift_limit=0.1, 
                                                        scale_limit=0.5)
        transformed_image = transform(image=image)['image']
    
    elif augmentation == 'smallest_max_size':
        transform = SmallestMaxSize(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'transpose':
        transform = Transpose(always_apply=True)
        transformed_image = transform(image=image)['image']

    elif augmentation == 'vertical_flip':
        transform = VerticalFlip(always_apply=True)
        transformed_image = transform(image=image)['image']

    name, ext = image_name.split('.')
    new_path = name + '_' + augmentation + '.' + ext
    cv2.imwrite(new_path, transformed_image)