import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from skimage import feature
from skimage.filters import gaussian


def show_SSIM_MSSIM(y_predicted,
                   y_test,
                   plot_image=False):
    predict_image = y_predicted.flatten()
    predict_image = predict_image.reshape(y_predicted.shape)
    predict_image = (predict_image).astype('float32')

    target_image = y_test.flatten()
    target_image = target_image.reshape(y_test.shape)
    target_image = target_image.astype('float32')

    SSIM = tf.image.ssim(target_image, predict_image, max_val=1).numpy()
    MSSIM = tf.image.ssim_multiscale(target_image, 
                                     predict_image, 
                                     max_val=1, 
                                     power_factors=(0.0448, 0.2856, 0.3001, 0.2363)).numpy()
    print('SSIM: ', SSIM)
    print('SSIM Multiscale: ', MSSIM)

    if plot_image:
        fig1, axes1 = plt.subplots(nrows=1, ncols=2)
        axes1[0].imshow(predict_image[275:,:,0])
        axes1[0].set_title("Predicted")
        axes1[1].imshow(target_image[275:,:,0])
        axes1[1].set_title("Target")
    
    return target_image, predict_image


def show_MSE_RMSE_MAE(target_image,
                      predict_image):
    mse_res = tf.keras.metrics.mean_squared_error(target_image.flatten(), predict_image.flatten()).numpy()
    mae_res = tf.keras.metrics.mean_absolute_error(target_image.flatten(), predict_image.flatten()).numpy()

    m = tf.keras.metrics.RootMeanSquaredError()
    m.update_state(target_image.flatten(), predict_image.flatten())
    
    return (mse_res, mae_res, m)


def soft_F_measure(target_data,
                   predicted_data,
                   canny_sigma=3,
                   gaussian_sigma=1,
                   beta_F1=1,
                   plot_data=False):
    e = feature.canny(target_data, sigma=canny_sigma)
    e_ = feature.canny(predicted_data, sigma=canny_sigma)

    e_ = gaussian(e_,sigma=gaussian_sigma,multichannel=None, preserve_range=True)

    TP = np.minimum(e,e_)
    TN = np.minimum(1-e,1-e_)
    FP = np.maximum(e_ - e , np.zeros(e.shape))
    FN = np.maximum(e - e_ , np.zeros(e.shape))
    sTP = np.sum(TP.flatten())
    sTN = np.sum(TN.flatten())
    sFP = np.sum(FP.flatten())
    sFN = np.sum(FN.flatten())
    
    soft_precision = sTP/(sTP+sFP)
    soft_recall = sTP/(sTP+sFN)
    
    soft_FM = (1+beta_F1**2)*(soft_precision*soft_recall)/(soft_precision*(beta_F1**2)+soft_recall)
    if plot_data:
    
        fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(16, 8))
        ax[0,0].imshow(target_data)
        ax[0,0].set_title('ground truth')

        ax[1,0].imshow(e, cmap='gray')
        ax[1,0].set_title('ground truth, canny, sigma:{}'.format(canny_sigma))

        ax[0,1].imshow(predicted_data)
        ax[0,1].set_title('predicted')

        ax[1,1].imshow(e_, cmap='gray')
        ax[1,1].set_title('predicted, canny, sigma:{}'.format(canny_sigma))
        
        ax[0,2].imshow(TP)
        ax[0,2].set_title('True Positive')

        ax[1,2].imshow(TN)
        ax[1,2].set_title('True Negative')

        ax[0,3].imshow(FP)
        ax[0,3].set_title('False Positive'.format(canny_sigma))

        ax[1,3].imshow(FN)
        ax[1,3].set_title('False Negative'.format(canny_sigma))

        fig.tight_layout()
        plt.show()
        
    return soft_FM